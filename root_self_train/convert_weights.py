"""Read NumPy-only Paddle checkpoints without importing Paddle; strict conversion."""
import argparse
import pickle
from pathlib import Path
import numpy as np


class ArrayReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module in {'numpy.core.multiarray', 'numpy._core.multiarray'} and name == '_reconstruct':
            from importlib import import_module
            return import_module('numpy._core.multiarray' if hasattr(np, '_core') else 'numpy.core.multiarray')._reconstruct
        if module == 'numpy' and name in {'ndarray', 'dtype'}:
            return getattr(np, name)
        raise ValueError(f'Unsupported checkpoint object: {module}.{name}')


def convert(source, destination):
    from .dependencies import ensure_imports
    ensure_imports(('torch',))
    import torch
    from .torch_model import SegFormerB2
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'不覆盖已有权重：{destination}')
    with open(source, 'rb') as stream:
        raw = ArrayReader(stream).load()
    raw.pop('StructuredToParameterName@@', None)
    model = SegFormerB2()
    linear = {name + '.weight' for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    converted = {}
    for name, array in raw.items():
        if not isinstance(array, np.ndarray) or not np.isfinite(array).all():
            raise ValueError(f'无效参数：{name}')
        target = name.replace('._mean', '.running_mean').replace('._variance', '.running_var')
        converted[target] = torch.from_numpy((array.T if name in linear else array).copy())
    converted['linear_fuse._batch_norm.num_batches_tracked'] = torch.tensor(0, dtype=torch.long)
    model.load_state_dict(converted, strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation ensures a failed concurrent conversion cannot overwrite weights.
    with destination.open('xb') as stream:
        torch.save(converted, stream)
    print(f'已严格转换 {len(raw)} 个参数/状态：{destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    convert(args.source, args.destination)
