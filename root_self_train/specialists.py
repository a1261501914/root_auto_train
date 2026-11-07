"""Local specialist agents. Qwen routes tasks; specialists own bounded execution.

Visual batch assessment can call Qwen when vision.enabled is configured.
"""
import json
from pathlib import Path

from .data import read_pairs, write_json


class Specialist:
    # 本地执行器的公共包装：先查数据/输出目录，再执行，并统一保存 agent_result.json。
    name = ''

    def run(self, cfg):
        from .agent import missing_data_guidance
        guidance = missing_data_guidance(self.action, cfg)
        if guidance:
            return dict(guidance, agent=self.name)
        if Path(cfg['workdir']).exists():
            raise ValueError(f"输出目录已存在，请指定新目录：{cfg['workdir']}")
        print(f'[{self.name}] 开始任务', flush=True)
        result = dict(self.perform(dict(cfg)), agent=self.name)
        write_json(Path(cfg['workdir']) / 'agent_result.json', result)
        print(f'[{self.name}] 任务完成', flush=True)
        return result


class SegmentationAgent(Specialist):
    # 此对象不再询问 LLM，而是直接调用已确定的图像分割流程。
    name, action = '分割智能体', 'predict'

    def perform(self, cfg):
        from . import __main__ as pipeline
        from .backend import Predictor
        if cfg.get('unlabeled_manifest'):
            manifest = Path(cfg['unlabeled_manifest'])
            images = [(manifest.parent / json.loads(line)['image']).resolve()
                      for line in manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        else:
            images = pipeline.input_images(cfg['unlabeled_dir'])
        if not images:
            raise ValueError('没有找到待分割图片，请检查输入目录或清单')
        predictor = Predictor(cfg['model_config'], cfg['checkpoint'], cfg['device'])
        rows = pipeline.predict_directory(predictor, images, Path(cfg['workdir']))
        return dict(status='completed', images=len(rows), output=cfg['workdir'],
                    checkpoint=cfg['checkpoint'], message='分割完成，尚未评估准确率')


class EvaluationAgent(Specialist):
    # 同一评估执行器的两条路径：assess_batch 无标签视觉质检；perform 有标签数值评估。
    name, action = '评估智能体', 'evaluate'

    def assess_batch(self, cfg):
        # quality.assess 会完成原图推理；vision.enabled=true 时再发送抽样图给视觉模型。
        from .quality import assess
        print('[评估智能体] 新批次评估：自动分割、本地风险检查及配置的视觉复核', flush=True)
        result = assess(cfg)
        return dict(result, agent=self.name, output=cfg['workdir'])

    def perform(self, cfg):
        # 这里的 Dice 来自人工标签，与视觉模型 good/risk/uncertain 的主观判断不同。
        from . import __main__ as pipeline
        from .backend import Predictor
        pairs = read_pairs(cfg['val_manifest'])
        predictor = Predictor(cfg['model_config'], cfg['checkpoint'], cfg['device'])
        detail = pipeline.evaluate_detailed(predictor, pairs)
        output = Path(cfg['workdir'])
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'evaluation.json', detail)
        pipeline.report_markdown(output / 'evaluation_report.md', '验证集评估', cfg['checkpoint'], detail)
        return dict(status='completed', metrics=detail['metrics'], output=str(output),
                    validation_target_met=detail['metrics']['dice'] >= cfg['auto_train_dice'],
                    message='仅评估固定人工验证集，不代表新批次逐图验收')


class TrainingAgent(Specialist):
    # 复用主流程的自训练闭环，再把最佳权重位置和是否晋级返回给调度器。
    name, action = '训练智能体', 'run'

    def perform(self, cfg):
        from . import __main__ as pipeline
        pipeline.run(cfg)
        best = json.loads((Path(cfg['workdir']) / 'best.json').read_text(encoding='utf-8'))
        return dict(status='completed', metrics=best['metrics'], checkpoint=best['checkpoint'],
                    output=cfg['workdir'], promoted=best['round'] > 0,
                    validation_target_met=best['metrics']['dice'] >= cfg['auto_train_dice'],
                    message='自训练结束，仅保留通过验证的模型')


class Coordinator:
    """Sequential handoffs avoid simultaneous GPU ownership and preserve task evidence."""
    def __init__(self):
        self.segmentation = SegmentationAgent()
        self.evaluation = EvaluationAgent()
        self.training = TrainingAgent()

    def run_auto(self, cfg):
        # 固定顺序：基线分割→人工集评估→必要时自训练→最终分割→人工集评估。
        # auto 当前使用人工验证指标来决定是否训练，不使用视觉复核结果作为训练门槛。
        from . import __main__ as pipeline
        pipeline.run(cfg, check=True)
        root = Path(cfg['workdir'])
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / 'config.json', cfg)
        trace = []

        def dispatch(agent, folder, checkpoint):
            # 每个阶段单独目录和日志；顺序执行，避免多个任务同时占用 GPU。
            task = dict(cfg, workdir=str(root / folder), checkpoint=checkpoint)
            entry = dict(agent=agent.name, task=folder, status='running')
            trace.append(entry)
            write_json(root / 'agent_trace.json', trace)
            try:
                result = agent.run(task)
                entry.update(status=result['status'], result=result)
                if result['status'] != 'completed':
                    raise ValueError(result['message'])
                return result
            except Exception:
                entry['status'] = 'failed'
                raise
            finally:
                write_json(root / 'agent_trace.json', trace)

        checkpoint = cfg['checkpoint']
        dispatch(self.segmentation, 'predictions_baseline', checkpoint)
        baseline = dispatch(self.evaluation, 'evaluation_baseline', checkpoint)
        status = 'accepted_baseline' if baseline['validation_target_met'] else 'below_target'
        if cfg['auto_train'] and not baseline['validation_target_met']:
            trained = dispatch(self.training, 'self_train', checkpoint)
            checkpoint = trained['checkpoint']
            status = 'improved' if trained['promoted'] else 'retained_baseline'
        dispatch(self.segmentation, 'predictions_final', checkpoint)
        final = dispatch(self.evaluation, 'evaluation_final', checkpoint)
        summary = dict(status=status, target_dice=cfg['auto_train_dice'], baseline=baseline['metrics'],
                       final=final['metrics'], checkpoint=checkpoint,
                       validation_target_met=final['validation_target_met'],
                       prediction_dir=str(root / 'predictions_final'),
                       evaluation=str(root / 'evaluation_final/evaluation_report.md'))
        write_json(root / 'auto_summary.json', summary)
        return summary
