"""读取数据清单、检查标签与集合隔离，并保存实验记录。"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def read_pairs(manifest):
    """读取 JSONL 图像/标签配对，相对路径以清单所在目录为基准。"""
    path = Path(manifest).resolve()
    result = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result.append(tuple((path.parent / row[k]).resolve() for k in ('image', 'mask')))
    if not result:
        raise ValueError(f'Empty manifest: {path}')
    return result


def digest(path):
    """对文件字节计算哈希；可识别改名副本，但不能识别重新编码的同一图像。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(train, val, unlabeled):
    """训练前校验尺寸、标签编码、损坏图片及集合内/集合间重复文件。"""
    groups = []
    root_count = 0
    for name, pairs in [('train', train), ('val', val)]:
        hashes = set()
        for image, mask in pairs:
            with Image.open(image) as im, Image.open(mask) as m:
                arr = np.asarray(m)
                if im.size != m.size or arr.ndim != 2:
                    raise ValueError(f'Mask must be single-channel and match image size: {mask}')
                if not set(np.unique(arr)).issubset({0, 1, 255}):
                    raise ValueError(f'Mask values must be 0/1/255: {mask}')
                if name == 'val':
                    root_count += int((arr == 1).sum())
            h = digest(image)
            if h in hashes:
                raise ValueError(f'Duplicate image in {name}: {image}')
            hashes.add(h)
        groups.append(hashes)
    # 验证集必须实际包含类别 1，防止把旧版 0/255 根标签当作忽略区域。
    if not root_count:
        raise ValueError('Validation masks contain no root pixels (class 1); check 0/255 encoding')
    hashes = set()
    for image in unlabeled:
        with Image.open(image) as im:
            im.verify()
        h = digest(image)
        if h in hashes:
            raise ValueError(f'Duplicate unlabeled image: {image}')
        hashes.add(h)
    groups.append(hashes)
    # 三组哈希两两求交集，避免已知的完全重复图片造成验证数据泄漏。
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Data leakage: identical image content crosses train/val/unlabeled splits')


def write_json(path, value):
    """先写同目录临时文件再替换，降低中断时留下半份结果的风险。"""
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)
