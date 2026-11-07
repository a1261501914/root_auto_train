"""Local, no-reference batch quality screening. These signals are not accuracy."""
import csv
import html
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

from .data import write_json


DEFAULTS = dict(min_root_pixels=20, min_root_confidence=.9, min_flip_dice=.9,
                max_uncertain_fraction=.2, uncertainty_low=.3, uncertainty_high=.7,
                small_component_pixels=20, max_small_component_fraction=.1,
                max_root_fraction=.5, thin_radius=2., min_thin_flip_recall=.85)
LABELS = {'passed_screening': '通过自动筛查', 'risk_detected': '发现风险',
          'insufficient_evidence': '证据不足', 'error': '检查失败'}
LIMITATION = '无人工真值：本报告不计算准确率或真实 Dice/IoU；通过筛查不代表分割正确。翻转一致性可能同时稳定地预测错误。'
ADVICE = {
    'low_confidence': '根区域置信度低：可能存在低对比度或场景差异，优先检查预处理与成像条件。',
    'flip_disagreement': '翻转前后不一致：可能对局部纹理或位置敏感，尝试切块尺度或推理增强并复测。',
    'uncertain_foreground': '候选根区域不确定像素多：可能存在边界模糊、细根漏分或背景混淆。',
    'small_fragments': '小连通区域占比偏高：可能有噪声或根系碎片；真实孤立细根也可能触发，勿直接删除。',
    'large_foreground': '预测根面积异常大：可能背景误分，也可能是真实密集根系。',
    'unstable_thin_regions': '细窄区域翻转后保留不足：提示细根或边缘不稳定，不能据此证明根系断裂。',
    'few_root_pixels': '预测根像素过少：无法区分真实空图与模型漏掉全部根系，不予判定通过。',
}


def settings(cfg):
    # 只允许已知阈值，避免配置拼写错误后被静默忽略；这些阈值是启发式规则，不是准确率。
    custom = cfg.get('quality', {})
    if not isinstance(custom, dict) or set(custom) - set(DEFAULTS):
        raise ValueError('quality 配置包含未知字段')
    result = dict(DEFAULTS, **custom)
    for key, value in result.items():
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f'quality.{key} 必须为非负有限数')
    for key in ('min_root_pixels', 'small_component_pixels'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'quality.{key} 必须为正整数')
    for key in set(result) - {'min_root_pixels', 'small_component_pixels', 'thin_radius'}:
        if result[key] > 1:
            raise ValueError(f'quality.{key} 必须在 0 到 1 之间')
    if not result['uncertainty_low'] < .5 < result['uncertainty_high']:
        raise ValueError('不确定概率区间必须跨越 0.5')
    return result


def inspect_prediction(prob, flipped, rules):
    # 输入是同一原图坐标下的两份 H×W 概率图；输出含指标记录和可视化所需的布尔掩膜。
    import cv2
    p, q = np.asarray(prob), np.asarray(flipped)
    if p.ndim != 2 or p.shape != q.shape or p.size == 0:
        raise ValueError('预测必须是同尺寸二维概率图')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or min(p.min(), q.min()) < 0 or max(p.max(), q.max()) > 1:
        raise ValueError('预测概率包含无效值')
    root, other = p >= .5, q >= .5
    # 0.5 将概率转为前景/背景；两次预测差异用于提示不稳定性，不能当作真实错误率。
    n, m = int(root.sum()), int(other.sum())
    uncertain = ((p > rules['uncertainty_low']) & (p < rules['uncertainty_high'])) | ((q > rules['uncertainty_low']) & (q < rules['uncertainty_high']))
    candidate = root | other | uncertain
    # Denominator focuses on potential roots, so abundant confident background cannot hide uncertainty.
    uncertainty = float(uncertain.sum() / max(1, int(candidate.sum())))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(root.astype('uint8'), connectivity=8)
    # 连通域就是互相连接的前景区域；小碎片过多可能是噪声，也可能是真实的孤立细根。
    sizes = stats[1:, cv2.CC_STAT_AREA]
    small = np.flatnonzero(sizes < rules['small_component_pixels']) + 1
    small_mask = np.isin(labels, small)
    small_fraction = float(sizes[sizes < rules['small_component_pixels']].sum() / max(1, n))
    # Distance-to-background measures narrow pixels, including boundaries of thick roots.
    radius = cv2.distanceTransform(np.pad(root.astype('uint8'), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    thin = root & (radius <= rules['thin_radius'])
    # 距离背景近的根像素被视为“细窄区域”；这也包含粗根边缘，不等同于细根骨架。
    thin_count = int(thin.sum())
    thin_recall = float((thin & other).sum()/thin_count) if thin_count else None
    score = dict(root_pixels=n, root_fraction=float(root.mean()),
                 root_confidence=float(p[root].mean()) if n else None,
                 flip_dice=float(2*(root & other).sum()/(n+m)) if n+m else None,
                 uncertain_candidate_fraction=uncertainty, component_count=count-1,
                 small_component_fraction=small_fraction, thin_pixels=thin_count,
                 thin_flip_recall=thin_recall)
    reasons = []
    if n < rules['min_root_pixels']:
        reasons.append('few_root_pixels')
    if n and score['root_confidence'] < rules['min_root_confidence']:
        reasons.append('low_confidence')
    if n+m and score['flip_dice'] < rules['min_flip_dice']:
        reasons.append('flip_disagreement')
    if uncertainty > rules['max_uncertain_fraction']:
        reasons.append('uncertain_foreground')
    if small_fraction > rules['max_small_component_fraction']:
        reasons.append('small_fragments')
    if score['root_fraction'] > rules['max_root_fraction']:
        reasons.append('large_foreground')
    if thin_count and thin_recall < rules['min_thin_flip_recall']:
        reasons.append('unstable_thin_regions')
    risks = [r for r in reasons if r != 'few_root_pixels']
    # 空预测缺少证据，不能因为“没有不稳定区域”而判为通过。
    status = 'risk_detected' if risks else ('insufficient_evidence' if reasons else 'passed_screening')
    return dict(status=status, metrics=score, reasons=reasons, suggestions=[ADVICE[r] for r in reasons]), dict(
        root=root, uncertain=uncertain, disagreement=root ^ other, fragments=small_mask)


def write_reports(output, rows, rules, checkpoint):
    # 本地指标报告：JSON 供程序读取，CSV 便于表格检查，HTML/Markdown 给用户阅读。
    counts = Counter(row['status'] for row in rows)
    # 优先报告风险；即使其他图片通过，只要有失败/空预测，也不能宣布整个批次通过。
    status = ('risk_detected' if counts['risk_detected'] else
              'insufficient_evidence' if counts['insufficient_evidence'] or counts['error'] else 'passed_screening')
    summary = dict(status=status, conclusion=LABELS[status], limitation=LIMITATION,
                   total=len(rows), counts=dict(counts), thresholds=rules, checkpoint=checkpoint,
                   complete=not counts['error'], report=str(output/'quality_report.html'))
    write_json(output/'quality_summary.json', summary)
    write_json(output/'quality_samples.json', rows)
    review = [row for row in rows if row['status'] != 'passed_screening']
    write_json(output/'review_queue.json', review)
    # Deterministic random sample distinct from risk selection, never a ground-truth estimate.
    random_indices = np.random.default_rng(42).choice(len(rows), min(10, len(rows)), replace=False)
    write_json(output/'random_sample.json', [rows[int(i)] for i in random_indices])
    with (output/'quality_samples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image','status','reasons',*next((list(r['metrics']) for r in rows if 'metrics' in r), [])], extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, reasons=';'.join(row.get('reasons',[])), **row.get('metrics',{})))
    cards = []
    lines = ['# 新批次质量筛查', '', summary['conclusion'], '', LIMITATION, '', f'共 {len(rows)} 张；状态统计：{dict(counts)}', '']
    for row in sorted(rows, key=lambda r: r['status'] == 'passed_screening'):
        # 风险图排在前面方便检查；写入 HTML 的动态文本要转义，避免被浏览器当标签执行。
        lines += [f"## {Path(row['image']).name}", LABELS[row['status']], *row.get('suggestions',[]), row.get('error',''), '']
        pictures = ''.join(f'<figure><img loading="lazy" src="{html.escape(row[key], quote=True)}"><figcaption>{label}</figcaption></figure>'
                           for key,label in [('preview','原图'),('overlay','红色：预测根系'),('risk_map','黄：不确定；紫：翻转分歧；青：小碎片')]
                           if key in row)
        description = '；'.join(row.get('suggestions',[])) or row.get('error', '未触发当前阈值；不等于准确率合格')
        cards.append(f"<article><h2>{html.escape(Path(row['image']).name)} · {LABELS[row['status']]}</h2><p>{html.escape(description)}</p><div>{pictures}</div><pre>{html.escape(json.dumps(row.get('metrics',{}), ensure_ascii=False, indent=2))}</pre></article>")
    (output/'quality_report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'quality_report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>根系质量筛查</title><style>body{font-family:system-ui;max-width:1200px;margin:auto;padding:24px;background:#f3f6f4}article{background:white;padding:20px;margin:16px 0}article div{display:flex;flex-wrap:wrap}figure{margin:8px;width:30%}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>新批次：'+summary['conclusion']+'</h1><p>'+LIMITATION+'</p><p>'+html.escape(str(dict(counts)))+'</p>'+''.join(cards)+'</html>', encoding='utf-8')
    return summary


def assess(cfg, predictor=None):
    # 无标签评估总入口。predictor 参数允许测试替换成假模型，不必真的加载 GPU 权重。
    from .__main__ import input_images
    from .dependencies import ensure_imports
    from .vision_quality import settings as vision_settings
    vision = vision_settings(cfg)
    output = Path(cfg['workdir'])
    if output.exists():
        raise ValueError(f'输出目录已存在，请更换：{output}')
    rules = settings(cfg)
    if cfg.get('unlabeled_manifest'):
        # 显式清单优先于扫描目录，可保证待处理批次固定且排除无关图片。
        manifest = Path(cfg['unlabeled_manifest'])
        images = [(manifest.parent/json.loads(line)['image']).resolve()
                  for line in manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    else:
        images = input_images(cfg['unlabeled_dir'])
    if not images:
        raise ValueError('没有找到图片，请指定待分割原图目录（不要混入标签或旧掩膜）')
    ensure_imports(('cv2',))
    if predictor is None:
        from .backend import Predictor
        predictor = Predictor(cfg['model_config'], cfg['checkpoint'], cfg['device'])
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('masks','probability','previews','overlays','risks'):
        (output/folder).mkdir()
    write_json(output/'quality_config.json', dict(thresholds=rules, checkpoint=cfg['checkpoint']))
    rows = []
    for index, image in enumerate(images):
        # 每张单独捕获图片/预测数据错误，保留失败记录，让其余图片继续完成检查。
        print(f'[质量检查] {index+1}/{len(images)} {image.name}', flush=True)
        try:
            with Image.open(image) as original:
                rgb = np.array(original.convert('RGB'))
            p, q = predictor.predict(image), predictor.predict(image, flip=True)
            # 原图推理看当前结果，翻转推理检查稳定性；predict() 内部已将 q 翻回原坐标。
            if p.shape != rgb.shape[:2]:
                raise ValueError('预测尺寸与原图不一致')
            row, maps = inspect_prediction(p, q, rules)
            stem = f'{index:06d}'  # Unique even if subdirectories contain identical names.
            Image.fromarray(maps['root'].astype('uint8')*255).save(output/'masks'/f'{stem}.png')
            Image.fromarray(np.clip(p*255,0,255).astype('uint8')).save(output/'probability'/f'{stem}.png')
            overlay = rgb.copy()
            # 只在副本叠加颜色，原始输入图像不改动；红色代表模型预测，而不是人工真值。
            overlay[maps['root']] = (.55*overlay[maps['root']]+.45*np.array([255,40,40])).astype('uint8')
            risk = rgb.copy()
            for name,color in [('uncertain',[255,220,0]),('disagreement',[255,0,255]),('fragments',[0,255,255])]:
                risk[maps[name]] = (.3*risk[maps[name]]+.7*np.array(color)).astype('uint8')
            paths = {}
            for key,folder,array in [('preview','previews',rgb),('overlay','overlays',overlay),('risk_map','risks',risk)]:
                preview = Image.fromarray(array)
                preview.thumbnail((1200,1200))
                paths[key] = f'{folder}/{stem}.jpg'
                preview.save(output/paths[key], quality=90)
            row.update(image=str(image), mask=f'masks/{stem}.png', **paths)
        except (OSError, ValueError) as exc:
            row = dict(image=str(image), status='error', error=str(exc), reasons=[])
        rows.append(row)
        write_json(output/'quality_samples.json', rows)
    summary = write_reports(output, rows, rules, cfg['checkpoint'])
    # 先保存全部本地结果，再做可选远程视觉复核；接口失败也能保留原有分割证据。
    if vision['enabled']:
        from .vision_quality import assess_visual
        visual = assess_visual(cfg, rows, summary)
        # 对用户优先反馈视觉总体报告，同时保留 local_report 和 local_conclusion 便于追溯。
        summary.update(vision=visual, report=visual['report'], message=visual['conclusion'],
                       local_report=str(output/'quality_report.html'),
                       local_conclusion=summary['conclusion'], conclusion=visual['conclusion'],
                       limitation=LIMITATION + ' ' + visual['limitation'], complete=visual['complete'])
        write_json(output/'quality_summary.json', summary)
        print('[评估反馈] ' + visual['conclusion'] + '\n总体报告：' + visual['report'], flush=True)
    return summary
