"""命令行入口与自训练闭环：检查 → 伪标签 → 微调 → 验证 → 晋级。"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from .dependencies import ensure_imports
try:
    ensure_imports(('numpy', 'yaml', 'PIL'))
except RuntimeError as exc:
    print(f'环境准备未完成：{exc}', file=sys.stderr)
    raise SystemExit(1) from None
import numpy as np
import yaml
from PIL import Image
from .core import counts, metrics, select
from .data import read_pairs, validate, write_json, digest

# 阅读导航：main() 分发命令；load() 读取配置；run() 管自训练；auto() 管组合任务。
# 实际神经网络计算放在 backend.py，这里主要负责把各个步骤串起来。


def load(path):
    """解析配置、统一绝对路径，并在启动训练前检查关键参数范围。"""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    # cfg 是贯穿流程的配置字典。路径相对于配置文件，而非相对于启动命令所在目录。
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
    # pairs 为人工图像/标签对。全局指标先累计像素计数再计算，不平均每张图片的分数。
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
        # 概率图适合看置信程度；二值掩膜适合后续处理；叠加图和缩略图用于人工查看。
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
    # 第 1 步：准备三组数据。训练标签用于学习，验证标签用于模型选择，未标注图用于伪标签。
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
    # Teacher 不是另一种网络，而是“当前保留的权重”；Student 是从它微调得到的候选权重。
    predictor = Predictor(cfg['model_config'], teacher, cfg['device'])
    # 先用相同验证流程建立 Teacher 基线，后续学生均与保留模型比较。
    score = evaluate(predictor, val)
    history = []
    write_json(work / 'best.json', dict(checkpoint=teacher, metrics=score, round=0))
    for r in range(1, cfg['rounds']+1):
        # 第 2 步：当前 Teacher 做原图/翻转两次推理，筛选可靠伪标签。
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
        # 第 3 步：人工样本 + 本轮通过筛选的伪标签样本，组成 Student 的训练数据。
        # 每轮重新生成伪标签，不累加旧轮标签；人工样本始终保留。
        pairs = train + accepted
        if any('\t' in str(p) or '\n' in str(p) for pair in pairs for p in pair):
            raise ValueError('Tabs and newlines in paths are unsupported')
        # 使用制表符分隔绝对路径，使文件名中的普通空格不影响解析。
        manifest.write_text(''.join(f'{a}\t{b}\n' for a,b in pairs), encoding='utf-8')
        del predictor
        import gc
        gc.collect()
        if cfg['device'].startswith(('gpu', 'cuda')):
            import torch
            torch.cuda.empty_cache()
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
        candidate = folder / 'student' / f"iter_{cfg['iters']}" / 'model.pt'
        # 第 4 步：使用同一份人工验证集比较新旧权重，避免用模型自己的伪标签为自己打分。
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
    """Coordinate segmentation, evaluation and training specialists locally."""
    from .specialists import Coordinator
    summary = Coordinator().run_auto(cfg)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    """check 只检查数据，run 启动闭环，worker 是内部单轮训练入口。"""
    parser = argparse.ArgumentParser(description='Root segmentation automatic prediction and self-training')
    parser.add_argument('command', choices=['check', 'predict', 'assess', 'evaluate', 'auto', 'run', 'worker', 'ask'])
    parser.add_argument('config')
    parser.add_argument('instruction', nargs='?', help='ask 的自然语言指令；省略时进入连续对话')
    parser.add_argument('--dry-run', action='store_true', help='ask 只解析并显示计划，不加载分割模型')
    parser.add_argument('--input-dir', help='image directory for predict; defaults to config.unlabeled_dir')
    parser.add_argument('--output', help='output directory; must not already exist')
    parser.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    # ask 先走自然语言规划；其他命令直接调用本地业务，不需要让 LLM 决定做什么。
    if args.command != 'ask' and (args.instruction is not None or args.dry_run):
        parser.error('instruction 和 --dry-run 仅适用于 ask')
    if args.command == 'ask':
        if args.input_dir or args.output or args.threshold != .5:
            parser.error('ask 的路径和要求请写入自然语言指令，不使用 --input-dir/--output/--threshold')
        from .agent import ask, chat
        try:
            if args.instruction is None:
                chat(args.config, dry_run=args.dry_run)
            else:
                ask(args.config, args.instruction, dry_run=args.dry_run)
        except (ValueError, RuntimeError, OSError) as exc:
            parser.exit(1, f'任务失败：{exc}\n')
        return
    if args.command == 'worker':
        # 内部训练子进程入口：这里的 config 参数实际上是上层生成的 job.json 文件路径。
        import json
        from .backend import train
        job = json.loads(Path(args.config).read_text(encoding='utf-8'))
        cfg = job['cfg']
        train(cfg['model_config'], job['teacher'], job['output'], job['manifest'], cfg)
    else:
        cfg = load(args.config)
        if args.command == 'assess':
            # 新批次评估：不读取人工验证标签；quality.py 负责筛查并按配置调用视觉复核。
            from .quality import assess
            if args.threshold != .5:
                parser.error('assess 使用固定二值阈值 0.5；质量阈值在 quality 配置中设置')
            if args.input_dir:
                cfg['unlabeled_dir'] = str(Path(args.input_dir).resolve())
                cfg.pop('unlabeled_manifest', None)
            if args.output:
                cfg['workdir'] = str(Path(args.output).resolve())
            try:
                summary = assess(cfg)
            except (ValueError, RuntimeError, OSError) as exc:
                parser.exit(1, f'质量检查未完成：{exc}\n')
            print(summary['conclusion'] + '：' + summary['limitation'])
            print('报告：' + summary['report'])
            return
        if args.command == 'check':
            run(cfg, check=True)
            return
        if args.command == 'auto':
            auto(cfg)
            return
        if args.command in ('predict', 'evaluate'):
            # evaluate 与 assess 含义不同：前者有人工真值，可算真实的分割重合指标。
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
