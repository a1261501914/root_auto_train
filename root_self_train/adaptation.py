"""视觉分数驱动的一次有界适配：评估→低分时自训练→同批次重新评估。"""
import json
import subprocess
from pathlib import Path
from .data import write_json
from .vision_quality import settings


def adapt(cfg):
    from .quality import assess
    from .specialists import TrainingAgent
    options = settings(cfg)
    if not options['enabled']:
        raise ValueError('adapt 需要 vision.enabled: true')
    root = Path(cfg['workdir'])
    root.mkdir(parents=True, exist_ok=False)
    baseline = assess(dict(cfg, workdir=str(root/'baseline')))
    scoring = baseline['vision']['scoring']
    result = dict(status='not_trained', baseline=baseline, checkpoint=cfg['checkpoint'],
                  report=baseline['report'], training=None)
    if scoring['decision'] == 'insufficient_evidence':
        result.update(status='needs_review', message='评分证据不足，未触发训练；请检查失败图片或增加可评估样本。')
    elif not scoring['training_recommended']:
        result['message'] = '视觉主观分达到阈值，未触发训练；不代表真实准确率达标。'
    elif not cfg.get('auto_train', False):
        result['message'] = '视觉主观分低于阈值，但 auto_train=false，未启动训练。'
    else:
        # 复用已有严格筛选伪标签和人工验证机制；不把低分掩膜直接当作可靠标签。
        try:
            training = TrainingAgent().run(dict(cfg, workdir=str(root/'training')))
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            result.update(status='training_failed', message='训练未完成，保留原模型：'+str(exc))
        else:
            result['training'] = training
            if training['status'] != 'completed':
                result.update(status=training['status'], message=training['message'])
            else:
                # 固定同一批次和 seed，使用训练流程保留的权重复评，不无限循环训练。
                result['checkpoint'] = training['checkpoint']
                final = assess(dict(cfg, checkpoint=training['checkpoint'], workdir=str(root/'final')))
                result.update(status='completed', final=final, report=final['report'],
                              message='训练流程已结束并重新视觉评估；权重晋级仍由人工验证集决定。')
    write_json(root/'adapt_summary.json', result)
    # 总入口报告可直接跳转到评分证据，并显示是否真的执行了训练。
    lines = ['# 视觉评分驱动适配报告', '', result['message'], '',
             f'状态：{result["status"]}', f'基线视觉主观分：{scoring["mean_score"]}；阈值：{scoring["threshold"]}',
             '[基线评估报告](baseline/vision_report.html)', '']
    if 'final' in result:
        lines += [f'最终视觉主观分：{result["final"]["vision"]["scoring"]["mean_score"]}',
                  '[最终评估报告](final/vision_report.html)']
    (root/'adapt_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result
