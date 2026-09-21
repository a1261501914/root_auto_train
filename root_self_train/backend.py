"""PaddleSeg 2.8 适配层；仅在真实推理或训练时导入 Paddle。"""
from pathlib import Path
import numpy as np
from PIL import Image


class Predictor:
    """加载完整分割权重，并输出恢复至原图尺寸的根系概率图。"""
    def __init__(self, config, checkpoint, device):
        import paddle
        from paddleseg.cvlibs import Config, SegBuilder
        from paddleseg.transforms import Compose
        from paddleseg.cvlibs import manager
        self.paddle = paddle
        paddle.set_device(device)
        cfg = Config(config)
        self.test_config = cfg.dic.get('test_config', {})
        def clear_pretrained(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'pretrained':
                        obj[key] = None
                    else:
                        clear_pretrained(value)
            elif isinstance(obj, list):
                for value in obj:
                    clear_pretrained(value)
        # 已有完整 Teacher 权重，禁用配置中的预训练加载，避免额外下载或覆盖。
        clear_pretrained(cfg.dic['model'])
        self.model = SegBuilder(cfg).model
        state = paddle.load(str(checkpoint))
        expected = self.model.state_dict()
        # 严格核对参数名与形状，防止配置不匹配时仅加载部分权重而继续运行。
        if set(state) != set(expected) or any(list(state[k].shape) != list(expected[k].shape) for k in expected):
            raise ValueError('Checkpoint keys/shapes do not exactly match the model config')
        self.model.set_state_dict(state)
        self.model.eval()
        transforms = cfg.dic['val_dataset']['transforms']
        # 当前通过缩放恢复原尺寸，尚不能逆转裁剪或填充，因此限制验证变换。
        allowed = {'Resize', 'ResizeByLong', 'LimitLong', 'Normalize'}
        if any(t['type'] not in allowed for t in transforms):
            raise ValueError('MVP validation transforms support only Resize/ResizeByLong/LimitLong/Normalize')
        self.transform = Compose([manager.TRANSFORMS[t['type']](**{k:v for k,v in t.items() if k != 'type'}) for t in transforms])

    def predict(self, image, flip=False):
        """返回 H×W 根概率；flip=True 时先翻图推理，再将概率图翻回。"""
        import cv2
        p = self.paddle
        arr = cv2.imread(str(image))
        if arr is None:
            raise ValueError(f'Cannot read image: {image}')
        height, width = arr.shape[:2]
        if flip:
            arr = arr[:, ::-1].copy()
        # OpenCV 读取 BGR，Compose 默认转 RGB，并沿用原配置的归一化。
        data = self.transform({'img': arr.astype('float32')})
        with p.no_grad():
            # 增加 batch 维度得到 NCHW，取模型输出列表中的主分割 logits。
            if self.test_config.get('is_slide', False):
                from .tiling import slide_logits
                def predict_tile(tile):
                    logits = self.model(p.to_tensor(tile[None].copy()))[0]
                    logits = p.nn.functional.interpolate(logits, size=list(tile.shape[1:]),
                                                         mode='bilinear', align_corners=False)
                    return logits[0].numpy()
                logits = slide_logits(data['img'], predict_tile,
                                      self.test_config['crop_size'], self.test_config['stride'])
                # 大图拼接和 softmax 留在 CPU，避免整幅双通道图长期占用显存。
                margin = logits[1] - logits[0]
                prob = np.exp(-np.logaddexp(0, -margin))
                if prob.shape != (height, width):
                    # 本配置只有 Normalize，不会走此分支；缩放配置需在 logits 上恢复尺寸。
                    logits = np.stack([cv2.resize(v, (width, height)) for v in logits])
                    prob = np.exp(-np.logaddexp(0, logits[0]-logits[1]))
                return prob[:, ::-1].copy() if flip else prob
            output = self.model(p.to_tensor(data['img'][None]))[0]
            if output.shape[1] != 2:
                raise ValueError('MVP requires two output channels: background=0, root=1')
            # 先恢复 logits 空间尺寸，再沿类别维 softmax，取类别 1（根系）。
            output = p.nn.functional.interpolate(output, size=[height, width], mode='bilinear', align_corners=False)
            prob = p.nn.functional.softmax(output, axis=1)[0, 1].numpy()
        return prob[:, ::-1].copy() if flip else prob


def train(config, checkpoint, output, manifest, options):
    """单轮微调：只继承 Teacher 参数，优化器和训练步数重新初始化。"""
    import paddle
    from paddleseg.cvlibs import Config, SegBuilder
    from paddleseg.core import train as seg_train
    from paddleseg.utils import utils
    utils.set_seed(options['seed'])
    predictor = Predictor(config, checkpoint, options['device'])
    cfg = Config(config, learning_rate=options['learning_rate'],
                 iters=options['iters'], batch_size=options['batch_size'])
    if cfg.dic['train_dataset']['type'] != 'Dataset':
        raise ValueError('MVP requires the generic PaddleSeg Dataset for training')
    # 替换为本轮人工数据与伪标签清单，保留原配置增强；255 为忽略像素。
    cfg.dic['train_dataset'].update(type='Dataset', dataset_root=str(Path(manifest).parent),
        train_path=str(manifest), num_classes=2, ignore_index=255, mode='train', separator='\t')
    def clear(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'pretrained':
                    obj[k] = None
                else:
                    clear(v)
        elif isinstance(obj, list):
            for v in obj:
                clear(v)
    clear(cfg.dic['model'])
    builder = SegBuilder(cfg)
    # 在构建优化器前加载 Teacher；不传 resume_model，不恢复旧优化器状态。
    builder.model.set_state_dict(predictor.model.state_dict())
    del predictor
    seg_train(builder.model, builder.train_dataset, optimizer=builder.optimizer,
              losses=builder.loss, save_dir=str(output), iters=options['iters'],
              batch_size=options['batch_size'], num_workers=0,
              save_interval=options['iters'], log_iters=10, keep_checkpoint_max=1)
    # 基础版统一评估最后一步 checkpoint，由外层固定验证集决定是否晋级。
    checkpoint = Path(output) / f"iter_{options['iters']}" / 'model.pdparams'
    if not checkpoint.is_file():
        raise RuntimeError(f'Training did not produce expected checkpoint: {checkpoint}')
    return checkpoint
