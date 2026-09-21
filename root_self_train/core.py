"""与深度学习框架无关的伪标签筛选和二分类分割指标。"""
import numpy as np


def counts(pred, target):
    """统计根系像素的 TP、FP、FN；标签 255 不参与任何计数。"""
    pred, target = np.asarray(pred, bool), np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError('Prediction and mask shapes differ')
    valid = target != 255
    truth = target == 1
    return np.array([np.sum(pred & truth & valid),
                     np.sum(pred & ~truth & valid),
                     np.sum(~pred & truth & valid)], dtype=np.int64)


def metrics(total):
    """由累计计数计算指标；预测和真值都无根时 Dice/IoU 约定为 1。"""
    tp, fp, fn = map(int, total)
    return dict(dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0,
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.0,
                precision=tp/(tp+fp) if tp+fp else 0.0,
                recall=tp/(tp+fn) if tp+fn else 0.0)


def select(prob, flipped_prob, cfg):
    """筛选单张图，返回 0/1/255 掩码及质量记录。

    两张输入均为原图坐标下的根系概率；翻转预测必须先翻回原方向。
    accepted 表示通过图像质量门槛，是否入训练集还取决于本轮数量上限。
    """
    p, q = np.asarray(prob), np.asarray(flipped_prob)
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError('Expected two equally shaped H x W probability maps')
    if not (np.isfinite(p).all() and np.isfinite(q).all()) or min(p.min(), q.min()) < 0 or max(p.max(), q.max()) > 1:
        raise ValueError('Invalid probability map')
    root, other = p >= .5, q >= .5
    n = int(root.sum())
    # 仅统计预测根区域，避免大量高置信背景掩盖根系预测错误。
    confidence = float(p[root].mean()) if n else 0.0
    # 用两次二值预测的 Dice 衡量翻转一致性；空根预测不会因此获得高分。
    consistency = float(2*(root & other).sum()/max(1, n+int(other.sum())))
    # 两次预测都对同一类别有足够信心才赋标签，其余像素保持忽略值。
    mask = np.full(p.shape, 255, dtype=np.uint8)
    mask[(p <= 1-cfg['pixel_confidence']) & (q <= 1-cfg['pixel_confidence'])] = 0
    mask[(p >= cfg['pixel_confidence']) & (q >= cfg['pixel_confidence'])] = 1
    coverage = float((mask != 255).mean())
    # 同时控制根像素数量、根置信度、一致性及可用于训练的像素比例。
    accepted = bool(n >= cfg['min_root_pixels'] and (mask == 1).any()
                    and confidence >= cfg['root_confidence']
                    and consistency >= cfg['consistency'] and coverage >= cfg['min_coverage'])
    return mask, dict(accepted=accepted, root_pixels=n, root_confidence=confidence,
                     consistency=consistency, coverage=coverage)
