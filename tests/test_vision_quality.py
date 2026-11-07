import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from PIL import Image
from root_self_train.vision_quality import settings, review_image, assess_visual


REVIEW = dict(verdict='risk', summary='右侧可见疑似漏分', missed_roots='右侧细根',
              false_positives='未见明显误分', thin_roots='局部不清楚', boundaries='需复核', suggestions=['核对右侧'])


class VisionTests(unittest.TestCase):
    def test_configuration_validation(self):
        self.assertFalse(settings({})['enabled'])
        for value in ({'enabled': 'true'}, {'max_images': 0}, {'max_side': 9000}, {'model': ''}):
            with self.assertRaises(ValueError):
                settings({'vision': value})

    def test_payload_and_invalid_provider_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'image.png'
            Image.new('RGB', (32, 32)).save(image)
            row = dict(image=str(image), mask='image.png', overlay='image.png')
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(
                {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(REVIEW)}}]}).encode()
            with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'secret'}), patch(
                    'root_self_train.vision_quality.urlopen', return_value=response) as remote:
                self.assertEqual(review_image({}, root, row, settings({})), REVIEW)
                request = remote.call_args.args[0]
                payload = json.loads(request.data)
                self.assertEqual(payload['model'], 'qwen3.8-flash')
                self.assertEqual(len(payload['messages'][1]['content']), 4)
                self.assertNotIn(str(root), request.data.decode())
                response.__enter__.return_value.read.return_value = b'{"choices": []}'
                with self.assertRaises(RuntimeError):
                    review_image({}, root, row, settings({}))

    def test_sampling_partial_failure_and_report_escaping(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [dict(image=f'{i}.png', status='passed_screening', preview=f'{i}.jpg', overlay=f'{i}.jpg') for i in range(5)]
            cfg = dict(workdir=tmp, vision={'enabled': True, 'max_images': 2})
            local = dict(conclusion='通过自动筛查', complete=True, counts={})
            with patch('root_self_train.vision_quality.review_image', side_effect=[dict(REVIEW, summary='<script>bad</script>'), RuntimeError('超时')]):
                summary = assess_visual(cfg, rows, local)
            self.assertEqual(summary['reviewed'], 1)
            self.assertEqual(summary['selected'], 2)
            self.assertFalse(summary['complete'])
            self.assertFalse(summary['full_coverage'])
            self.assertIn('风险', summary['conclusion'])
            report = (Path(tmp)/'vision_report.html').read_text(encoding='utf-8')
            self.assertNotIn('<script>', report)
            self.assertIn('&lt;script&gt;', report)

    def test_all_failures_and_empty_batch_never_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(workdir=tmp, vision={'enabled': True})
            local = dict(conclusion='通过自动筛查', complete=True, counts={})
            with patch('root_self_train.vision_quality.review_image', side_effect=RuntimeError('失败')):
                summary = assess_visual(cfg, [dict(image='x', status='passed_screening', preview='x', overlay='x')], local)
            self.assertIn('证据不足', summary['conclusion'])
            self.assertFalse(summary['complete'])
            self.assertIn('证据不足', assess_visual(cfg, [], local)['conclusion'])
