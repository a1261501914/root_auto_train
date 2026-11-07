"""PyTorch inference and fine-tuning; no Paddle runtime dependency."""
import random
from pathlib import Path
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageOps


def select_device(torch, device):
    # 配置里可写 gpu:0；PyTorch 实际使用的名称是 cuda:0。
    name = 'cuda' + str(device)[3:] if str(device).startswith('gpu') else str(device)
    resolved = torch.device(name)
    if resolved.type not in {'cpu', 'cuda'}:
        raise ValueError('设备仅支持 cpu、gpu/cuda 或 gpu:0/cuda:0')
    if resolved.type == 'cuda' and (not torch.cuda.is_available() or (resolved.index or 0) >= torch.cuda.device_count()):
        raise RuntimeError('当前 PyTorch 无法使用指定 GPU，请检查 CUDA 版 PyTorch 和驱动；可显式设置 device: cpu')
    return resolved


class Predictor:
    # 将“加载网络、预处理、推理”封装为一个对象，供分割、评估和伪标签生成复用。
    def __init__(self, config, checkpoint, device):
        if Path(checkpoint).suffix == '.pdparams':
            raise ValueError('请先转换旧权重：python -m root_self_train.convert_weights model/model.pdparams model/model.pt；然后修改 checkpoint 为 model/model.pt')
        from .dependencies import ensure_imports
        ensure_imports(('torch',), device)
        import torch
        from .torch_model import SegFormerB2
        self.torch, self.device = torch, select_device(torch, device)
        self.cfg = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
        model = self.cfg['model']
        if model['type'] != 'SegFormer' or model['backbone']['type'] != 'MixVisionTransformer_B2' or model['num_classes'] != 2:
            raise ValueError('当前 PyTorch 后端仅支持二分类 SegFormer-B2')
        transforms = self.cfg['val_dataset']['transforms']
        if len(transforms) != 1 or transforms[0]['type'] != 'Normalize':
            raise ValueError('PyTorch 验证变换当前仅支持 Normalize')
        norm = transforms[0]
        self.mean = np.array(norm.get('mean', [.5]*3), dtype=np.float32)[:, None, None]
        self.std = np.array(norm.get('std', [.5]*3), dtype=np.float32)[:, None, None]
        if self.mean.shape != (3, 1, 1) or self.std.shape != (3, 1, 1) or not (self.std > 0).all():
            raise ValueError('Normalize mean/std 必须为三个通道，std 必须大于零')
        self.test_config = self.cfg.get('test_config', {})
        self.model = SegFormerB2(model['embedding_dim'])
        # 权重先在 CPU 读取，再整体移动到目标设备；strict=True 禁止缺层时继续使用随机参数。
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.model.to(self.device).eval()

    def normalize(self, image):
        # 图像 H×W×3 → 模型要求的 3×H×W；先缩放到 [0,1]，再按通道标准化。
        return (np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255 - self.mean) / self.std

    def predict(self, image, flip=False):
        # 返回 H×W 的“根系概率”，不是二值图片；调用方决定如何阈值化。
        from .tiling import slide_logits
        torch = self.torch
        with Image.open(image) as source:
            arr = self.normalize(source.convert('RGB'))
        if flip:
            arr = arr[:, :, ::-1].copy()
        def tile(chw):
            # 增加 batch 维：CHW → 1×C×H×W。小图补到至少 32，输出时裁回原尺寸。
            h, w = chw.shape[1:]
            x = torch.from_numpy(chw.copy()).unsqueeze(0).to(self.device)
            x = torch.nn.functional.pad(x, (0, max(0, 32-w), 0, max(0, 32-h)))
            return self.model(x)[0, :, :h, :w].cpu().numpy()
        with torch.inference_mode():
            # 推理不建立反向传播计算图；滑窗模式逐块预测，降低 GPU 峰值占用。
            if self.test_config.get('is_slide', False):
                logits = slide_logits(arr, tile, self.test_config['crop_size'], self.test_config['stride'])
            else:
                logits = tile(arr)
        prob = np.exp(-np.logaddexp(0, logits[0] - logits[1]))
        # 上式等价于两类 softmax 的根系通道概率，采用数值稳定形式避免指数溢出。
        # 翻转推理后必须翻回原图坐标，否则无法逐像素比较两次预测。
        return prob[:, ::-1].copy() if flip else prob


def training_sample(image, mask, transforms, normalize):
    # 图像和标签必须使用相同的几何变换；只改变图像颜色，不改变标签类别。
    with Image.open(image) as source, Image.open(mask) as target:
        image, mask = source.convert('RGB'), target.copy()
    for transform in transforms:
        kind = transform['type']
        if kind == 'ResizeStepScaling':
            lo, hi, step = transform['min_scale_factor'], transform['max_scale_factor'], transform.get('scale_step_size', .25)
            scale = random.choice(np.arange(lo, hi + 1e-6, step)) if step else random.uniform(lo, hi)
            size = tuple(max(1, int(v * scale)) for v in image.size)
            # 标签用最近邻缩放，避免插值制造出不存在的类别值。
            image, mask = image.resize(size, Image.Resampling.BILINEAR), mask.resize(size, Image.Resampling.NEAREST)
        elif kind == 'RandomPaddingCrop':
            w, h = transform['crop_size']
            pad = (0, 0, max(0, w-image.width), max(0, h-image.height))
            # 补出来的标签填 255（忽略），避免把没有真实内容的区域当作背景学习。
            image, mask = ImageOps.expand(image, pad, fill=(127, 127, 127)), ImageOps.expand(mask, pad, fill=255)
            x, y = random.randint(0, image.width-w), random.randint(0, image.height-h)
            box = (x, y, x+w, y+h)
            image, mask = image.crop(box), mask.crop(box)
        elif kind == 'RandomHorizontalFlip':
            if random.random() < transform.get('prob', .5):
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
        elif kind == 'RandomDistort':
            for key, enhancement in [('brightness_range', ImageEnhance.Brightness), ('contrast_range', ImageEnhance.Contrast), ('saturation_range', ImageEnhance.Color)]:
                magnitude = transform.get(key, 0)
                image = enhancement(image).enhance(random.uniform(1-magnitude, 1+magnitude))
        elif kind != 'Normalize':
            raise ValueError(f'不支持训练变换：{kind}')
    return normalize(image), np.array(mask, dtype=np.int64)


def train(config, checkpoint, output, manifest, options):
    # 从 Teacher 权重继续微调，而非从随机网络开始；每轮都创建新的优化器。
    predictor = Predictor(config, checkpoint, options['device'])
    torch, model = predictor.torch, predictor.model
    seed = options['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    pairs = [line.split('\t') for line in Path(manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError('训练清单必须包含制表符分隔的图像与掩膜路径')
    transforms = predictor.cfg['train_dataset']['transforms']
    optimizer_config = predictor.cfg['optimizer']
    scheduler = predictor.cfg['lr_scheduler']
    loss_config = predictor.cfg['loss']
    if optimizer_config['type'] != 'AdamW' or scheduler['type'] != 'PolynomialDecay' or scheduler.get('power', 1) != 1:
        raise ValueError('当前训练仅支持 AdamW + PolynomialDecay(power=1)')
    if loss_config != {'types': [{'type': 'CrossEntropyLoss'}], 'coef': [1]}:
        raise ValueError('当前训练仅支持单个 CrossEntropyLoss，ignore_index=255')
    norms = [item for item in transforms if item['type'] == 'Normalize']
    if norms != predictor.cfg['val_dataset']['transforms'] or transforms[-1]['type'] != 'Normalize':
        raise ValueError('训练需以与验证相同的 Normalize 结束')
    model.train()
    # train() 开启 Dropout 和 BatchNorm 的训练行为；它与是否计算梯度是两回事。
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'],
                                 betas=(optimizer_config.get('beta1', .9), optimizer_config.get('beta2', .999)),
                                 weight_decay=optimizer_config.get('weight_decay', .01))
    for step in range(options['iters']):
        # 每步有放回随机抽取 batch_size 对样本，因此这里按步数训练，不按 epoch 遍历。
        samples = [training_sample(*random.choice(pairs), transforms, predictor.normalize) for _ in range(options['batch_size'])]
        images = torch.from_numpy(np.stack([s[0] for s in samples])).to(predictor.device)
        masks = torch.from_numpy(np.stack([s[1] for s in samples])).to(predictor.device)
        if not (masks != 255).any():
            raise ValueError('训练 batch 全为忽略像素，请检查标签或裁剪配置')
        optimizer.zero_grad(set_to_none=True)
        # 清空旧梯度 → 计算逐像素交叉熵 → 反向传播 → 更新参数。
        # 伪标签中不可靠的像素标为 255，不参与损失计算。
        loss = torch.nn.functional.cross_entropy(model(images), masks, ignore_index=255)
        if not torch.isfinite(loss):
            raise RuntimeError('训练损失非有限值，停止并保留原模型')
        loss.backward()
        optimizer.step()
        for group in optimizer.param_groups:
            # power=1 的 PolynomialDecay：学习率从初始值线性下降到 0。
            group['lr'] = options['learning_rate'] * (1 - (step+1)/options['iters'])
        if step % 10 == 0 or step + 1 == options['iters']:
            print(f'iter={step+1} loss={loss.item():.6f}', flush=True)
    destination = Path(output) / f"iter_{options['iters']}" / 'model.pt'
    # 只保存参数，不保存整个 Python 模型对象；xb 模式禁止覆盖已有权重。
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, stream)
    return destination
