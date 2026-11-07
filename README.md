# 根系分割自主训练：PyTorch + Deep Agents

当前分割和训练后端为原生 PyTorch SegFormer-B2，不需要 Paddle/PaddleSeg，也不依赖 Transformers。
千问通过 LangChain Deep Agents 主控委派分割、评估、训练三个 LLM 子智能体；计划经本地校验后执行。

## 启动

建议使用 Python 3.11–3.13，Deep Agents 要求 >=3.11。

```powershell
pip install -r requirements.txt -r requirements-agent.txt
# 仅首次创建；已有配置不要覆盖
Copy-Item config.example.yaml config.yaml
$env:DASHSCOPE_API_KEY = "你的百炼 API Key"
python -m root_self_train ask config.yaml
```

缺少已知依赖时，先显示缺失项、目标 Python 和安装命令，询问是否同意。
只有输入 `y`、`yes`、`是` 或 `同意` 才自动安装；回车、拒绝或无法读取输入均不安装，
反馈当前步骤检测到的缺失依赖和手动安装命令。对话模式保留会话，基础启动依赖缺失则说明原因后退出。
每个安装命令单独确认；安装成功后重试导入并继续任务，不重放已完成的训练。
PyTorch 安装固定 2.8.0：CPU 使用官方 cpu 源，
GPU 使用官方 cu128 源；安装后检查设备，不静默切到 CPU，不安装驱动或 CUDA Toolkit。
已有 torch 不自动覆盖。可用 `python -m root_self_train.environment gpu` 只读检查环境。
[PyTorch 官方安装命令](https://pytorch.org/get-started/previous-versions/)

## 模型与旧权重迁移

配置默认指向：

```yaml
model_config: configs/root_segformer_b2.yml
checkpoint: model/model.pt
device: gpu
agent_backend: deepagents
```

已有 `model/model.pdparams` 可以一次性转换，无需安装 Paddle：

```powershell
python -m root_self_train.convert_weights model/model.pdparams model/model.pt
```

转换器仅支持本项目 NumPy 格式的 SegFormer-B2 二分类权重；限制反序列化对象，
按 Linear 层定义转置参数（包括方阵），映射 BatchNorm 状态，严格匹配所有键及形状。
保留原文件，拒绝覆盖目标文件。PyTorch 加载使用 `weights_only=True`，不允许部分加载或随机权重回退。
本地已转换真实权重并完成 GPU 推理、一步微调和 checkpoint 重载验证；尚无 Paddle/PyTorch 逐像素数值对照。
其他模型的任意 `.pt` 文件不保证可加载，必须匹配本项目网络结构。

## 连续对话

```text
你好
只分割 E:/data/roots，不评估、不训练
评估当前模型在人工验证集上的效果
分割并评估，Dice 低于 0.85 就自训练，最多一轮，每轮 10 步
```

`exit`/`退出` 结束，`/clear` 清空上下文，`/result` 查看本地反馈。
附带一句指令只执行一次；`--dry-run` 调用模型规划但不执行分割训练。
记住最近 12 轮用户消息和模型回复，退出后不恢复。
Qwen 模型、兼容接口地址和 API Key 环境变量名可在 `qwen` 配置调整；模型须支持工具调用。
`QWEN_MODEL`、`QWEN_BASE_URL` 可覆盖配置；地域须与 Key 匹配。

Deep Agents 使用内存 StateBackend，不挂载本地文件或 shell，不启用 LangSmith 追踪。
规划子智能体仅提交任务计划，不读取执行结果。启用 `vision.enabled` 后，评估阶段会另行发送抽样原图、掩膜和叠加图给千问视觉模型；不发送模型权重或人工标签。当前仍非根据训练反馈反复自主规划的完整闭环。
复合任务由本地 Coordinator 按固定规则执行。`agent_backend: legacy` 可切回旧 JSON 路由。
[Deep Agents 子智能体文档](https://docs.langchain.com/oss/python/deepagents/subagents)

## 数据准备

## 新批次全自动质量筛查（不需要人工标签）

```powershell
python -m root_self_train assess config.yaml --input-dir "E:/data/new_roots" --output runs/quality_001
```

也可以在连续对话中输入：“自动分割并检查 E:/data/new_roots 的分割效果，不训练。”
评估子智能体将提交 `assess`，不会使用原来的 50 张人工验证图，且无需训练/验证清单。
请只提供原图目录，别把标签或旧预测掩膜混在里面；目前是重新分割后检查，不是导入任意现有掩膜评分。

程序完成两次推理（原图、水平翻转）、根区域置信度、候选根区域不确定比例、翻转一致性、
小连通域占比、根面积以及细窄区域稳定性检查。细窄像素按距离变换近似，包含粗根的边缘，
不是骨架根长、断裂检测或真实细根召回率。连通域使用原图像素，阈值受分辨率影响。

输出 `quality_report.html`（原图/叠加/风险图对照）、Markdown、JSON 和 CSV。
`review_queue.json` 自动汇总可疑图片及规则触发证据，`random_sample.json` 单独保存固定种子的随机抽样。
可疑样本选择用于定位问题，不能冒充整体准确率估计。原图不修改；输出文件按序号命名以避免同名覆盖。

结论分为“通过自动筛查”“发现风险”“证据不足”；空根预测不判为通过，坏图片记录为检查失败，
批次不会因部分图片失败而误报全部通过。阈值可在 `quality` 配置修改，默认规则未经你的新场景校准。
**无真值不能给出真实 Dice/IoU 或证明正确**；高置信、稳定的预测也可能错误。
因此报告不自动宣布精度达标，不伪造人工标签，也不根据这些启发式分数直接训练或发布模型。
以上启发式检查在本地完成。当前配置另外启用了视觉复核，见下文。

### 千问视觉评估与总体报告

`assess` 及评估子智能体支持视觉复核，当前 `config.yaml` 和示例配置使用：

```yaml
vision:
  enabled: true
  model: qwen3.8-flash
  max_images: 10
  max_side: 1600
```

复用 `qwen.base_url`、`qwen.api_key_env`、`qwen.timeout` 和 `QWEN_BASE_URL`；视觉模型由 `vision.model` 单独指定，不受 `QWEN_MODEL` 覆盖。
调用方式仍为上面的 `assess` 命令，或对话输入“用视觉模型评估这批图片的分割效果，生成总体报告，不训练”。
程序先完成本地分割和筛查，再以固定种子随机抽样，每张发送原图、二值掩膜及叠加图，最大边默认缩至 1600 像素。
这会向配置的远程接口上传图片并产生模型调用费用；设 `vision.enabled: false` 可仅执行本地检查，未配置 vision 的旧配置默认关闭视觉复核。

新增 `vision_report.html`、`vision_report.md`、`vision_summary.json`、`vision_samples.json`，反馈总体观感、漏分、误分、细根、边界与连续性及逐图复核建议。
报告显示总图片数、抽样数、成功数、失败数及覆盖范围；失败不会伪装为通过，本地筛查报告仍保留。
视觉评估是主观判断，缩图可能隐藏细根，不能给出真实 Dice/IoU、替代人工真值或触发自动训练／模型晋级。
`evaluate` 和 `auto` 中的人工验证集指标评估仍使用原来的数值流程。
接口图像输入格式参见 [阿里云视觉理解文档](https://www.alibabacloud.com/help/en/model-studio/vision/)。

## 人工标注数据准备

仅分割不需要人工标签。评估需要验证集，训练需要训练和验证集，自动流程当前要求两者都有。
缺少清单时显示如何准备数据及如何先仅分割，保持对话，不启动任务。
训练/验证 JSONL 每行：

```json
{"image":"images/001.png","mask":"masks/001.png"}
```

路径相对于清单文件。标签必须同尺寸、单通道，0=背景、1=根系、255=忽略。
训练/验证/未标注图片应分离；程序按文件字节哈希排查完全重复，近似图片仍需人工分组。
使用自带 9 对白色根标签进行联调：

```powershell
python prepare_local_data.py --unlabeled-dir "E:/data/roots" --device gpu
python -m root_self_train ask config.local.yaml
```

脚本生成 7 张训练、2 张验证的临时划分及 8 张未标注冒烟清单，不是最终独立测试集。

## 命令行与自动流程

```powershell
python -m root_self_train check config.yaml
python -m root_self_train predict config.yaml --input-dir "E:/data/roots" --output runs/predictions_001
python -m root_self_train evaluate config.yaml --output runs/evaluation_001
python -m root_self_train run config.yaml
python -m root_self_train auto config.yaml
```

预测输出原尺寸概率图、二值掩膜、叠加图、缩略图；支持 CPU 合并重叠 logits 的滑窗。
评估输出全局与逐图 Dice、IoU、Precision、Recall，仅代表固定人工验证集。
原图与水平翻转预测生成伪标签，用置信度、一致性、有效覆盖率筛选；默认伪标签数量不超过人工样本数。
每轮重新初始化优化器，使用 AdamW、线性衰减、交叉熵（忽略 255），最后一步保存 `student/iter_N/model.pt`。
学生 Dice 严格超过旧模型加 min_delta 才晋级，否则停止并保留旧模型。

当前验证变换只支持 Normalize。训练支持随机尺度、填充裁剪、水平翻转、亮度/对比度/饱和度扰动。
这些增强用 Pillow 实现，与旧 Paddle 增强不保证数值相同；训练轮数/步数/学习率取顶层任务配置。
模型配置中的优化器和损失目前仅支持所提供的 AdamW、PolynomialDecay(power=1)、CrossEntropyLoss 组合。
GPU 不可用时说明原因，可显式配置 `device: cpu`；别将成功转换当成精度已经验证。

每次使用新结果目录，不覆盖旧实验。`best.json`、`history.json`、`training.log` 保存模型选择与训练过程。
自动流程另保存 `agent_trace.json`、`auto_summary.json`；评估报告位于 `evaluation_baseline/`、`evaluation_final/`。

## 测试

```powershell
python -m unittest discover -s tests -v
```

包括指标、伪标签、调度、Deep Agents 离线委派、依赖安装模拟以及 PyTorch 前向/反向和权重重载。
缺少可选框架时相关测试跳过；测试不会自动安装 PyTorch。旧项目说明和过程记录保留为历史资料，以此 README 为准。
