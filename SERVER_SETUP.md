# 迁移到 GPU 电脑或服务器

## 需要传输

- `root_self_train/`、`tests/`、`prepare_local_data.py`、`requirements.txt`
- `configs/root_segformer_b2.yml`、`config.example.yaml`
- `root_get/`：原权重和配置
- `dataset/`：9 对人工图像及原始标签
- 未标注图片目录：可独立放在服务器的数据盘，不必放入项目

使用原来能训练此模型的 Paddle 环境。当前代码适配目标是 PaddleSeg 2.8 的 Config/SegBuilder；需要先确认服务器实际 Paddle/PaddleSeg 版本，再做真实权重联调。本机没有安装或更改服务器环境。

## 在项目根目录运行

```bash
pip install -r requirements.txt
python -c "import paddle,paddleseg; print(paddle.__version__); print(paddle.is_compiled_with_cuda())"
python prepare_local_data.py --unlabeled-dir /your/server/unlabeled --device gpu
python -m root_self_train check config.local.yaml
python -m root_self_train run config.local.yaml
```

把示例路径替换为实际未标注目录。Windows GPU 电脑同样使用 `--unlabeled-dir` 传入数据目录。
准备脚本会在目标机器生成绝对路径清单；不要直接复用本机生成的 `data/prepared/*.jsonl`，其中含本机路径。

第一轮配置只用前 8 张合格未标注图像、10 iter、batch_size=1，检查推理、伪标签、训练和评估接口。阈值严格时可能没有合格伪标签，流程会正常停止；此时查看质量记录，不应直接认为训练接口已验证。

联调完成后，把 `config.local.yaml` 中 `unlabeled_manifest` 改为 `data/prepared/unlabeled.jsonl`，按需要增加 iters/rounds，并为 `workdir` 使用新的目录名。准备脚本会重写联调配置，正式训练前建议另存配置。

当前验证集只有 080、605 两张，适合基础流程联调。正式指标应扩充按植株或采集批次隔离的验证集，并另留测试集。
