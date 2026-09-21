"""大图滑窗：在 CPU 累计重叠区域 logits，限制单次送入 GPU 的尺寸。"""
import numpy as np


def slide_logits(chw, predict_tile, crop_size, stride):
    """输入 CHW 图像，回调返回 2×H×W logits；尺寸参数顺序为宽、高。"""
    cw, ch = crop_size
    sw, sh = stride
    if min(cw, ch, sw, sh) <= 0 or sw > cw or sh > ch:
        raise ValueError('Sliding stride must be positive and no larger than crop')
    _, height, width = chw.shape
    total = np.zeros((2, height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    # 将末端窗口贴齐边缘，既覆盖非整除尺寸，也避免遗漏最后几行或列。
    for row in range(max(1, (max(height-ch, 0)+sh-1)//sh+1)):
        y = max(0, min(row*sh, height-ch))
        for col in range(max(1, (max(width-cw, 0)+sw-1)//sw+1)):
            x = max(0, min(col*sw, width-cw))
            tile = chw[:, y:y+ch, x:x+cw]
            logits = predict_tile(tile)
            if logits.shape != (2, *tile.shape[1:]):
                raise ValueError('Tile logits must have two classes and match tile size')
            h, w = tile.shape[1:]
            total[:, y:y+h, x:x+w] += logits
            count[y:y+h, x:x+w] += 1
    # 先平均 logits 再计算 softmax，与先平均概率的算法不同。
    return total / count[None]
