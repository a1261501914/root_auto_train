import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from root_self_train.specialists import Coordinator


class CoordinatorTests(unittest.TestCase):
    def test_handoffs_and_training_gate(self):
        for allow, score, expected in [(True, .7, True), (False, .7, False), (True, .9, False)]:
            with self.subTest(allow=allow, score=score), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'run'
                coordinator = Coordinator()
                cfg = dict(workdir=str(root), checkpoint='teacher', auto_train=allow, auto_train_dice=.85)
                evaluation = dict(status='completed', metrics={'dice': score}, validation_target_met=score >= .85)
                with patch('root_self_train.__main__.run'), \
                        patch.object(coordinator.segmentation, 'run', return_value={'status': 'completed'}) as seg, \
                        patch.object(coordinator.evaluation, 'run', return_value=evaluation), \
                        patch.object(coordinator.training, 'run', return_value=dict(status='completed', checkpoint='student', promoted=True)) as train:
                    result = coordinator.run_auto(cfg)
                self.assertEqual(train.called, expected)
                self.assertEqual(seg.call_args.args[0]['checkpoint'], 'student' if expected else 'teacher')
                self.assertEqual(result['validation_target_met'], score >= .85)
                trace = json.loads((root / 'agent_trace.json').read_text(encoding='utf-8'))
                self.assertEqual(len(trace), 5 if expected else 4)

    def test_failure_stops_downstream_and_records_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            coordinator = Coordinator()
            with patch('root_self_train.__main__.run'), \
                    patch.object(coordinator.segmentation, 'run', side_effect=RuntimeError('failed')), \
                    patch.object(coordinator.evaluation, 'run') as evaluation:
                with self.assertRaises(RuntimeError):
                    coordinator.run_auto(dict(workdir=str(root), checkpoint='teacher'))
            evaluation.assert_not_called()
            trace = json.loads((root / 'agent_trace.json').read_text(encoding='utf-8'))
            self.assertEqual(trace[0]['status'], 'failed')
