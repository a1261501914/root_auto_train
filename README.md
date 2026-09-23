# 根系分割自主训练 · 基础版

完整中文说明请参阅：[项目说明文档.md](项目说明文档.md)。

复用已有 **PaddleSeg SegFormer-B2 配置和 `.pdparams` 权重**，执行有限轮数的伪标签自训练。第一版为命令行程序。

已有 Teacher → 未标注图像预测与水平翻转预测 → 质量筛选 → 人工数据 + 伪标签 → Student 微调 → 固定验证集 Dice 晋级。

当前命令行闭环支持分步运行，也支持只执行一次 `auto` 完成全流程：

```bash
# 1. 预测并保存 masks、概率图、叠加图和缩略图
python -m root_self_train predict config.yaml --input-dir data/unlabeled --output runs/predictions_001

# 2. 用人工标注验证集评估，并生成 JSON 与 Markdown 报告
python -m root_self_train evaluate config.yaml --output runs/evaluation_001

# 3. 只有确认指标不足且人工数据已整理好后，才启动伪标签自训练
python -m root_self_train run config.yaml

# 推荐：一次启动，自动预测、评估、按 Dice 判断是否自训练，再输出最终结果
python -m root_self_train auto config.yaml
```

`predict` 和 `evaluate` 均严格使用配置中的 SegFormer-B2 YAML 与 checkpoint；`evaluate` 的晋级依据是固定验证集的全局 Dice，报告同时包含 IoU、Precision、Recall 和逐图结果。
`auto` 默认以 `auto_train_dice` 作为触发阈值：低于阈值才自训练，训练后仍必须在固定验证集上提升 `min_delta` 才会选用 Student；所有中间结果和最终结果都保存在 `workdir` 下。

## 已实现

- 原图分辨率概率图；根区域平均置信度、翻转 Dice、有效像素覆盖率筛选。
- 伪标签背景=0、根系=1、不确定=255（忽略）；空根预测不入选。
- 每轮伪标签数量上限，默认不超过人工样本数；按文件名顺序取合格样本。
- 每轮从 Teacher 权重初始化，重新建立优化器；沿用原配置损失和训练增强。
- 验证集全局根系 Dice/IoU/Precision/Recall；Dice 超过旧模型至少 `min_delta` 才晋级。
- 未晋级或无合格伪标签即停止，原始权重不覆盖；每次运行使用新目录。
- 图片内容哈希检查跨集合重复；记录配置、输入哈希、样本质量、训练日志和结果。

## 环境

优先在原有可运行 SegFormer-B2 的训练环境安装本项目依赖：

```bash
pip install -r requirements.txt
python -m root_self_train --help
```

适配目标为 PaddleSeg 2.8（`Config` + `SegBuilder`），PaddlePaddle 按已有 GPU/CUDA 环境安装。本项目没有自动安装或修改 GPU 框架。
API 对照：[官方训练入口](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/tools/train.py)、[图像变换实现](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/paddleseg/transforms/transforms.py)。

## 数据与配置

`train.jsonl` 和 `val.jsonl` 每行一个 JSON 对象，路径相对于各自清单文件，也可以使用绝对路径：

```json
{"image":"labeled/images/001.png","mask":"labeled/masks/001.png"}
{"image":"labeled/images/002.png","mask":"labeled/masks/002.png"}
```

训练集和验证集使用不同图片。未标注图像放入独立目录。人工 mask 必须为单通道、与图像同尺寸，取值 **0=背景、1=根系、255=忽略**。旧数据若用 255 表示根，需先转换成 1；程序不会猜测 255 的含义。

复制 `config.example.yaml` 为 `config.yaml`，填写原始模型 YAML、权重、两个清单和未标注目录。路径相对于此配置文件。Windows 路径建议用正斜杠。

原始模型配置应为二分类 SegFormer，使用 PaddleSeg 通用 `Dataset`。训练变换必须让每个 batch 尺寸一致，损失应支持 ignore_index=255。验证变换仅支持确定性的 Resize / ResizeByLong / LimitLong / Normalize，必须保留原训练使用的归一化参数。支持 `test_config.is_slide` 滑窗推理，按 `crop_size` 和 `stride` 分块并在 CPU 平均重叠 logits。验证裁剪、填充、自定义 Dataset 和多卡训练暂未适配。

```bash
# 只检查路径、图片、标签和数据泄漏，不加载 Paddle 或启动训练
python -m root_self_train check config.yaml

# 开始真实闭环（默认最多 3 轮，每轮 1000 iter）
python -m root_self_train run config.yaml
```

训练日志在 `runs/experiment_001/round_01/training.log`。`best.json` 指向当前保留权重；`history.json` 是各轮指标，`samples.json` 记录每个样本筛选结果，`hard_samples.json` 是待人工审核清单（不会自动提交标注）。学生使用该轮最后一个 checkpoint 进行评估。

验证集用于模型选择，最终报告应另用独立测试集；哈希只能发现完全相同的图片，同一植株的相邻帧需在划分时自行分组。阈值只是初始参数，不保证伪标签正确或指标提高。

## 验证状态

```bash
python -m unittest discover -s tests -v
```

包含指标与忽略像素、伪标签门控、晋级边界，以及模拟模型驱动的完整两轮编排测试。模拟测试没有训练神经网络。
当前交付未使用真实 SegFormer 权重或 GPU 执行训练，PaddleSeg 接口仍需在实际训练环境做一轮小规模联调。

下一步接入需要：原始模型 YAML（及其 `_base_` 文件）、`model.pdparams`、少量训练/验证/未标注图片和对应人工 mask，以及 Paddle/PaddleSeg/CUDA 版本。建议先把 `rounds` 设为 1、`iters` 设为 10 完成冒烟测试，再正式训练。
