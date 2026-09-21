"""命令行入口与自训练闭环：检查 → 伪标签 → 微调 → 验证 → 晋级。"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
from .core import counts, metrics, select
from .data import read_pairs, validate, write_json, digest


def load(path):
    """解析配置、统一绝对路径，并在启动训练前检查关键参数范围。"""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg.setdefault('auto_train', True)
    cfg.setdefault('auto_train_dice', .85)
    for key in ('model_config', 'checkpoint', 'train_manifest', 'val_manifest', 'unlabeled_dir', 'workdir'):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    if cfg.get('unlabeled_manifest'):
        cfg['unlabeled_manifest'] = str((path.parent / cfg['unlabeled_manifest']).resolve())
    for key in ('rounds', 'iters', 'batch_size', 'min_root_pixels'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('root_confidence', 'consistency', 'pixel_confidence', 'min_coverage'):
        if not 0 <= cfg[key] <= 1:
            raise ValueError(f'{key} must be between 0 and 1')
    if cfg['pixel_confidence'] <= .5:
        raise ValueError('pixel_confidence must exceed 0.5')
    for key in ('learning_rate', 'max_pseudo_ratio'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not math.isfinite(cfg['min_delta']) or cfg['min_delta'] < 0:
        raise ValueError('min_delta must be finite and nonnegative')
    if not isinstance(cfg['auto_train'], bool):
        raise ValueError('auto_train must be true or false')
    if not 0 <= cfg['auto_train_dice'] <= 1:
        raise ValueError('auto_train_dice must be between 0 and 1')
    return cfg


def evaluate(predictor, pairs):
    """累计整个验证集的 TP/FP/FN 后计算指标，而非逐图 Dice 的平均值。"""
    total = np.zeros(3, dtype=np.int64)
    for image, mask in pairs:
        with Image.open(mask) as im:
            total += counts(predictor.predict(image) >= .5, np.asarray(im))
    return metrics(total)


def evaluate_detailed(predictor, pairs, threshold=.5):
    """Return global metrics plus per-image metrics for an auditable report."""
    total = np.zeros(3, dtype=np.int64)
    rows = []
    for image, mask in pairs:
        with Image.open(mask) as im:
            target = np.asarray(im)
        prob = predictor.predict(image)
        item = counts(prob >= threshold, target)
        total += item
        rows.append(dict(image=str(image), mask=str(mask), **metrics(item),
                         tp=int(item[0]), fp=int(item[1]), fn=int(item[2])))
    return dict(metrics=metrics(total), threshold=threshold, samples=rows,
                counts=dict(tp=int(total[0]), fp=int(total[1]), fn=int(total[2])))


def input_images(directory):
    suffixes = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    return sorted(p.resolve() for p in Path(directory).rglob('*') if p.suffix.lower() in suffixes)


def predict_directory(predictor, images, output):
    """Export binary masks, probability maps, overlays and a machine-readable index."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    folders = {name: output / name for name in ('masks', 'probability', 'overlays', 'thumbnails')}
    for folder in folders.values():
        folder.mkdir()
    rows = []
    for image in images:
        with Image.open(image) as source:
            rgb = np.asarray(source.convert('RGB'))
        prob = predictor.predict(image)
        mask = (prob >= .5).astype(np.uint8) * 255
        stem = image.stem
        mask_path = folders['masks'] / f'{stem}_mask.png'
        prob_path = folders['probability'] / f'{stem}_prob.png'
        overlay_path = folders['overlays'] / f'{stem}_overlay.png'
        thumb_path = folders['thumbnails'] / f'{stem}_thumb.jpg'
        Image.fromarray(mask).save(mask_path)
        Image.fromarray(np.clip(prob * 255, 0, 255).astype(np.uint8)).save(prob_path)
        overlay = rgb.copy()
        overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
        Image.fromarray(overlay).save(overlay_path)
        Image.fromarray(overlay).thumbnail((640, 640))
        # thumbnail() mutates the PIL image; save a fresh copy to retain the full overlay.
        preview = Image.fromarray(overlay)
        preview.thumbnail((640, 640))
        preview.save(thumb_path, quality=90)
        rows.append(dict(image=str(image), mask=str(mask_path), probability=str(prob_path),
                         overlay=str(overlay_path), thumbnail=str(thumb_path),
                         root_pixels=int((mask > 0).sum()), max_probability=float(prob.max()),
                         mean_root_probability=float(prob[mask > 0].mean()) if (mask > 0).any() else 0.0))
    write_json(output / 'prediction_summary.json', rows)
    return rows


def report_markdown(path, title, checkpoint, detail):
    score = detail['metrics']
    lines = [f'# {title}', '', f'- checkpoint: `{checkpoint}`',
             f'- threshold: `{detail["threshold"]}`', f'- samples: `{len(detail["samples"])}`', '',
             '## Overall metrics', '', '| Dice | IoU | Precision | Recall |', '|---:|---:|---:|---:|',
             f'| {score["dice"]:.6f} | {score["iou"]:.6f} | {score["precision"]:.6f} | {score["recall"]:.6f} |', '',
             '## Per-image results', '', '| Image | Dice | IoU | Precision | Recall |', '|---|---:|---:|---:|---:|']
    lines += [f'| `{Path(row["image"]).name}` | {row["dice"]:.6f} | {row["iou"]:.6f} | {row["precision"]:.6f} | {row["recall"]:.6f} |'
              for row in detail['samples']]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def promoted(candidate, teacher, delta):
    """仅当学生 Dice 有限且严格超过 Teacher Dice + delta 时晋级。"""
    return math.isfinite(candidate['dice']) and candidate['dice'] > teacher['dice'] + delta


def run(cfg, check=False, workdir=None):
    """执行有限轮自训练；检查模式完成数据校验后立即返回，不加载模型。"""
    train, val = read_pairs(cfg['train_manifest']), read_pairs(cfg['val_manifest'])
    unlabeled = sorted(p.resolve() for p in Path(cfg['unlabeled_dir']).rglob('*')
                       if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'})
    # 显式清单可排除已标注图像的副本，无需复制体积较大的外部数据目录。
    if cfg.get('unlabeled_manifest'):
        manifest_path = Path(cfg['unlabeled_manifest'])
        unlabeled = [(manifest_path.parent / json.loads(line)['image']).resolve()
                     for line in manifest_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not unlabeled:
        raise ValueError('No unlabeled images found')
    validate(train, val, unlabeled)
    for key in ('model_config', 'checkpoint'):
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(cfg[key])
    print(f'Validated: train={len(train)}, val={len(val)}, unlabeled={len(unlabeled)}', flush=True)
    if check:
        return
    from .backend import Predictor
    work = Path(workdir or cfg['workdir'])
    # 禁止复用已有实验目录，避免把旧日志或旧 checkpoint 混入本次结果。
    work.mkdir(parents=True, exist_ok=False)
    write_json(work / 'config.json', cfg)
    write_json(work / 'input_hashes.json', {str(p): digest(p) for pair in train+val for p in pair} |
               {str(p): digest(p) for p in unlabeled} | {cfg['checkpoint']: digest(cfg['checkpoint'])})
    teacher = cfg['checkpoint']
    predictor = Predictor(cfg['model_config'], teacher, cfg['device'])
    # 先用相同验证流程建立 Teacher 基线，后续学生均与保留模型比较。
    score = evaluate(predictor, val)
    history = []
    write_json(work / 'best.json', dict(checkpoint=teacher, metrics=score, round=0))
    for r in range(1, cfg['rounds']+1):
        folder = work / f'round_{r:02d}'
        folder.mkdir()
        pseudo = folder / 'pseudo'
        pseudo.mkdir()
        rows, accepted = [], []
        # 限制伪标签相对人工数据的数量，防止噪声样本占据大部分训练集。
        limit = max(1, int(len(train)*cfg['max_pseudo_ratio']))
        for i, image in enumerate(unlabeled):
            mask, quality = select(predictor.predict(image), predictor.predict(image, flip=True), cfg)
            row = dict(image=str(image), **quality)
            if quality['accepted'] and len(accepted) < limit:
                destination = pseudo / f'{i:06d}.png'
                Image.fromarray(mask).save(destination)
                accepted.append((image, destination))
                row.update(selected=True, mask=str(destination))
            else:
                row.update(selected=False, reason='quota' if quality['accepted'] else 'quality')
            rows.append(row)
        write_json(folder / 'samples.json', rows)
        # 因数量上限未选入的合格样本不算难样本；仅收集质量不达标者。
        write_json(folder / 'hard_samples.json', [row for row in rows if not row['accepted']])
        if not accepted:
            write_json(folder / 'result.json', dict(status='stopped_no_pseudo_labels'))
            break
        manifest = folder / 'train.txt'
        # 每轮重新生成伪标签，不累加旧轮标签；人工样本始终保留。
        pairs = train + accepted
        if any('\t' in str(p) or '\n' in str(p) for pair in pairs for p in pair):
            raise ValueError('Tabs and newlines in paths are unsupported')
        # 使用制表符分隔绝对路径，使文件名中的普通空格不影响解析。
        manifest.write_text(''.join(f'{a}\t{b}\n' for a,b in pairs), encoding='utf-8')
        del predictor
        import gc
        gc.collect()
        if cfg['device'].startswith('gpu'):
            import paddle
            paddle.device.cuda.empty_cache()
        # 先释放父进程推理缓存；独立子进程训练，退出后释放优化器与显存。
        job = dict(cfg=cfg, teacher=str(teacher), manifest=str(manifest), output=str(folder / 'student'))
        write_json(folder / 'job.json', job)
        try:
            with (folder / 'training.log').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-m', 'root_self_train', 'worker', str(folder / 'job.json')],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            write_json(folder / 'result.json', dict(status='training_failed', retained_teacher=str(teacher)))
            raise
        candidate = folder / 'student' / f"iter_{cfg['iters']}" / 'model.pdparams'
        predictor = Predictor(cfg['model_config'], candidate, cfg['device'])
        new_score = evaluate(predictor, val)
        promote = promoted(new_score, score, cfg['min_delta'])
        result = dict(round=r, teacher_metrics=score, student_metrics=new_score,
                      promoted=promote, pseudo_count=len(accepted), student=str(candidate))
        # best.json 只指向通过验证的权重；学生失败或退步均不覆盖旧模型。
        if promote:
            teacher, score = str(candidate), new_score
            write_json(work / 'best.json', dict(checkpoint=teacher, metrics=score, round=r))
        history.append(result)
        write_json(folder / 'result.json', result)
        write_json(work / 'history.json', history)
        print(f'Round {r}: Dice={new_score["dice"]:.5f}, promoted={promote}', flush=True)
        # 未改善立即停止，避免相同 Teacher 与参数重复产生同样的实验。
        if not promote:
            break
    print(f'Best checkpoint: {teacher}', flush=True)


def auto(cfg):
    """One-command orchestrator: predict, evaluate, conditionally self-train, then finalize."""
    # Reuse run's validation logic without loading Paddle or creating an experiment directory.
    run(cfg, check=True)
    root = Path(cfg['workdir'])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'config.json', cfg)
    from .backend import Predictor
    images = input_images(cfg['unlabeled_dir'])
    if cfg.get('unlabeled_manifest'):
        manifest_path = Path(cfg['unlabeled_manifest'])
        images = [(manifest_path.parent / json.loads(line)['image']).resolve()
                  for line in manifest_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    predictor = Predictor(cfg['model_config'], cfg['checkpoint'], cfg['device'])
    baseline_prediction = root / 'predictions_baseline'
    predict_directory(predictor, images, baseline_prediction)
    val = read_pairs(cfg['val_manifest'])
    baseline = evaluate_detailed(predictor, val)
    write_json(root / 'evaluation_baseline.json', baseline)
    report_markdown(root / 'evaluation_baseline.md', 'Baseline evaluation', cfg['checkpoint'], baseline)
    target = cfg['auto_train_dice']
    need_train = cfg['auto_train'] and baseline['metrics']['dice'] < target
    final_checkpoint = cfg['checkpoint']
    status = 'accepted_baseline'
    if need_train:
        train_cfg = dict(cfg, workdir=str(root / 'self_train'))
        run(train_cfg)
        best = json.loads((root / 'self_train' / 'best.json').read_text(encoding='utf-8'))
        final_checkpoint = best['checkpoint']
        status = 'self_trained_and_selected'
    elif not cfg['auto_train']:
        status = 'baseline_below_target_training_disabled' if baseline['metrics']['dice'] < target else status
    del predictor
    predictor = Predictor(cfg['model_config'], final_checkpoint, cfg['device'])
    final_prediction = root / 'predictions_final'
    predict_directory(predictor, images, final_prediction)
    final = evaluate_detailed(predictor, val)
    write_json(root / 'evaluation_final.json', final)
    report_markdown(root / 'evaluation_final.md', 'Final evaluation', final_checkpoint, final)
    summary = dict(status=status, target_dice=target, baseline=baseline['metrics'],
                   final=final['metrics'], checkpoint=final_checkpoint,
                   prediction_dir=str(final_prediction), evaluation=str(root / 'evaluation_final.md'))
    write_json(root / 'auto_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    """check 只检查数据，run 启动闭环，worker 是内部单轮训练入口。"""
    parser = argparse.ArgumentParser(description='Root segmentation automatic prediction and self-training')
    parser.add_argument('command', choices=['check', 'predict', 'evaluate', 'auto', 'run', 'worker'])
    parser.add_argument('config')
    parser.add_argument('--input-dir', help='image directory for predict; defaults to config.unlabeled_dir')
    parser.add_argument('--output', help='output directory; must not already exist')
    parser.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    if args.command == 'worker':
        import json
        from .backend import train
        job = json.loads(Path(args.config).read_text(encoding='utf-8'))
        cfg = job['cfg']
        train(cfg['model_config'], job['teacher'], job['output'], job['manifest'], cfg)
    else:
        cfg = load(args.config)
        if args.command == 'check':
            run(cfg, check=True)
            return
        if args.command == 'auto':
            auto(cfg)
            return
        if args.command in ('predict', 'evaluate'):
            from .backend import Predictor
            if not 0 < args.threshold < 1:
                raise ValueError('--threshold must be between 0 and 1')
            predictor = Predictor(cfg['model_config'], cfg['checkpoint'], cfg['device'])
            if args.command == 'predict':
                source = args.input_dir or cfg['unlabeled_dir']
                images = input_images(source)
                if not images:
                    raise ValueError(f'No images found: {source}')
                output = args.output or str(Path(cfg['workdir']).parent / 'predictions')
                rows = predict_directory(predictor, images, output)
                print(f'Predicted {len(rows)} images. Results: {output}', flush=True)
            else:
                detail = evaluate_detailed(predictor, read_pairs(cfg['val_manifest']), args.threshold)
                output = Path(args.output or (Path(cfg['workdir']).parent / 'evaluation'))
                output.mkdir(parents=True, exist_ok=False)
                write_json(output / 'evaluation.json', detail)
                report_markdown(output / 'evaluation_report.md', 'SegFormer-B2 evaluation',
                                cfg['checkpoint'], detail)
                print(json.dumps(detail['metrics'], ensure_ascii=False), flush=True)
                print(f'Report: {output / "evaluation_report.md"}', flush=True)
        else:
            run(cfg)


if __name__ == '__main__':
    main()
