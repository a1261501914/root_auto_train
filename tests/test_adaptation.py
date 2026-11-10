import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from root_self_train.adaptation import adapt
from root_self_train.agent import validate_plan


class AdaptTests(unittest.TestCase):
    def test_gate_training_and_reevaluation(self):
        for decision, enabled, calls in [('below_threshold', True, 1), ('below_threshold', False, 0),
                                         ('meets_threshold', True, 0), ('insufficient_evidence', True, 0)]:
            with self.subTest(decision=decision, enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                cfg = dict(workdir=str(Path(tmp)/'adapt'), checkpoint='teacher.pt', auto_train=enabled,
                           seed=42, vision={'enabled': True})
                summary = dict(report='report.html', vision={'scoring': dict(mean_score=70, threshold=80,
                               decision=decision, training_recommended=decision == 'below_threshold')})
                with patch('root_self_train.quality.assess', return_value=summary) as assess, patch(
                        'root_self_train.specialists.TrainingAgent.run', return_value=dict(status='completed', checkpoint='best.pt')) as train:
                    result = adapt(cfg)
                self.assertEqual(train.call_count, calls)
                self.assertEqual(assess.call_count, 1+calls)
                if calls:
                    self.assertEqual(assess.call_args.args[0]['checkpoint'], 'best.pt')
                    self.assertEqual(result['checkpoint'], 'best.pt')
                self.assertTrue((Path(cfg['workdir'])/'adapt_summary.json').exists())

    def test_missing_training_data_keeps_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(workdir=str(Path(tmp)/'adapt'), checkpoint='teacher.pt', auto_train=True, vision={'enabled': True})
            baseline = dict(report='baseline.html', vision={'scoring': dict(mean_score=20, threshold=80,
                            decision='below_threshold', training_recommended=True)})
            with patch('root_self_train.quality.assess', return_value=baseline) as assess, patch(
                    'root_self_train.specialists.TrainingAgent.run', return_value=dict(status='needs_setup', message='缺少人工标签')):
                result = adapt(cfg)
            self.assertEqual(result['status'], 'needs_setup')
            self.assertEqual(result['checkpoint'], 'teacher.pt')
            self.assertEqual(assess.call_count, 1)

    def test_plan_requires_explicit_training_flag(self):
        plan = dict(action='adapt', message='视觉低分则训练', options={})
        with self.assertRaises(ValueError):
            validate_plan(plan, '', {})
        plan['options']['auto_train'] = True
        self.assertEqual(validate_plan(plan, '', {})['action'], 'adapt')
