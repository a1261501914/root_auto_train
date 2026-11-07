"""Native PyTorch MiT-B2 + SegFormer head, matching the existing parameter layout."""
import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    # 重叠卷积提取局部图块特征，再展开为 token 序列，保留 h、w 供后续还原空间布局。
    def __init__(self, inputs, channels, kernel, stride):
        super().__init__()
        self.proj = nn.Conv2d(inputs, channels, kernel, stride, kernel // 2)
        self.norm = nn.LayerNorm(channels, eps=1e-5)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), h, w


class Attention(nn.Module):
    # 多头注意力：Q 表示当前位置的查询，K/V 提供其他位置的信息。
    # reduction > 1 时只压缩 K/V 的空间分辨率，减少注意力矩阵的计算量。
    def __init__(self, channels, heads, reduction):
        super().__init__()
        self.heads, self.reduction = heads, reduction
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, channels * 2)
        self.proj = nn.Linear(channels, channels)
        if reduction > 1:
            self.sr = nn.Conv2d(channels, channels, reduction, reduction)
            self.norm = nn.LayerNorm(channels, eps=1e-5)

    def forward(self, x, h, w):
        b, n, c = x.shape
        # b=batch 数，n=h*w 个空间位置，c=通道数；每个头处理 c/heads 个通道。
        q = self.q(x).reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        reduced = x
        if self.reduction > 1:
            reduced = self.sr(x.transpose(1, 2).reshape(b, c, h, w))
            reduced = self.norm(reduced.flatten(2).transpose(1, 2))
        kv = self.kv(reduced).reshape(b, -1, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attention = (q @ kv[0].transpose(-2, -1) * (c // self.heads) ** -.5).softmax(-1)
        return self.proj((attention @ kv[1]).transpose(1, 2).reshape(b, n, c))


class DWConv(nn.Module):
    # 逐通道 3×3 卷积：将 token 还原成特征图，让相邻像素交换局部信息。
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x, h, w):
        b, _, c = x.shape
        return self.dwconv(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    # 通道先扩展为四倍，再经局部卷积和 GELU 激活，最后投影回原通道数。
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels * 4)
        self.dwconv = DWConv(channels * 4)
        self.fc2 = nn.Linear(channels * 4, channels)

    def forward(self, x, h, w):
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), h, w)))


class Block(nn.Module):
    # 一个 Transformer 单元：归一化→注意力→残差，再归一化→MLP→残差。
    def __init__(self, channels, heads, reduction, drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.attn = Attention(channels, heads, reduction)
        self.mlp = MLP(channels)
        self.drop = drop

    def drop_path(self, x):
        # 训练时随机跳过部分样本的残差分支以减轻过拟合；评估时保持完整网络。
        if not self.training or not self.drop:
            return x
        keep = 1 - self.drop
        mask = (torch.rand((x.shape[0], 1, 1), device=x.device) < keep).to(x.dtype)
        return x * mask / keep

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        return x + self.drop_path(self.mlp(self.norm2(x), h, w))


class Backbone(nn.Module):
    # MiT-B2 编码器的四个阶段：分辨率逐级降低、通道逐级增加，输出四种尺度特征。
    def __init__(self):
        super().__init__()
        dims, depths = [64, 128, 320, 512], [3, 4, 6, 3]
        # dims 是各阶段通道数；depths 是各阶段堆叠的 Transformer 单元数量。
        offset = 0
        for i, (dim, depth, heads, reduction) in enumerate(zip(dims, depths, [1, 2, 5, 8], [8, 4, 2, 1])):
            setattr(self, f'patch_embed{i+1}', PatchEmbed(3 if i == 0 else dims[i-1], dim, 7 if i == 0 else 3, 4 if i == 0 else 2))
            setattr(self, f'block{i+1}', nn.ModuleList([Block(dim, heads, reduction, .1 * (offset+j) / 15) for j in range(depth)]))
            setattr(self, f'norm{i+1}', nn.LayerNorm(dim, eps=1e-6))
            offset += depth

    def forward(self, x):
        features = []
        for i in range(1, 5):
            tokens, h, w = getattr(self, f'patch_embed{i}')(x)
            for block in getattr(self, f'block{i}'):
                tokens = block(tokens, h, w)
            tokens = getattr(self, f'norm{i}')(tokens)
            x = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, h, w)
            features.append(x)
        return features


class Projection(nn.Module):
    # 四个尺度原本通道数不同，这里统一到 embedding，便于后面的拼接与融合。
    def __init__(self, channels, embedding):
        super().__init__()
        self.proj = nn.Linear(channels, embedding)

    def forward(self, x):
        b, _, h, w = x.shape
        return self.proj(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, -1, h, w)


class Fuse(nn.Module):
    # 1×1 卷积将拼接后的四份特征融合为一份，再用 BatchNorm 和 ReLU 稳定表示。
    def __init__(self, embedding):
        super().__init__()
        self._conv = nn.Conv2d(embedding * 4, embedding, 1, bias=False)
        self._batch_norm = nn.BatchNorm2d(embedding, eps=1e-5, momentum=.1)

    def forward(self, x):
        return F.relu(self._batch_norm(self._conv(x)))


class SegFormerB2(nn.Module):
    # 完整分割网络 = 编码器 Backbone + 多尺度投影/融合 + 两类像素分类头。
    def __init__(self, embedding=768):
        super().__init__()
        self.backbone = Backbone()
        for i, channels in enumerate([64, 128, 320, 512], 1):
            setattr(self, f'linear_c{i}', Projection(channels, embedding))
        self.linear_fuse = Fuse(embedding)
        self.dropout = nn.Dropout2d(.1)
        self.linear_pred = nn.Conv2d(embedding, 2, 1)

    def forward(self, x):
        features = self.backbone(x)
        # 将四个尺度都上采样到第一阶段的尺寸后融合；最后恢复到输入图像大小。
        # 输出 N×2×H×W 的 logits（未归一化分类分数），此处还不是概率或掩膜。
        projected = [F.interpolate(getattr(self, f'linear_c{i+1}')(f), size=features[0].shape[-2:],
                                   mode='bilinear', align_corners=False) for i, f in enumerate(features)]
        logits = self.linear_pred(self.dropout(self.linear_fuse(torch.cat(projected[::-1], dim=1))))
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
