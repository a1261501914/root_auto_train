# PyTorch 服务器接入

使用 Python 3.11–3.13，安装 requirements.txt 和 requirements-agent.txt。
将 model/model.pt 与模型配置一并复制到服务器；若只有旧权重，运行：

```bash
python -m root_self_train.convert_weights model/model.pdparams model/model.pt
python prepare_local_data.py --unlabeled-dir /data/roots --device gpu
python -m root_self_train check config.local.yaml
python -m root_self_train run config.local.yaml
```

PyTorch 按配置自动安装 CPU 或 CUDA 12.8 wheel；需要兼容 NVIDIA 驱动，不再安装 Paddle/PaddleSeg。
显式 CPU 模式使用 device: cpu。已有环境中的 torch 不会自动覆盖。
先用生成的 1 轮、10 iter 配置联调，伪标签为空时可能正常停止，不能视为训练已验证。
详细配置、Deep Agents 对话与数据格式见 README.md。
