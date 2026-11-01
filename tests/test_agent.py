import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from root_self_train.agent import ask, chat, execute, request_legacy_plan as request_plan, validate_plan, missing_data_guidance


class AgentTests(unittest.TestCase):
    def test_missing_manifests_give_guidance_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(self.cfg, train_manifest=str(Path(tmp) / 'train.jsonl'),
                       val_manifest=str(Path(tmp) / 'val.jsonl'))
            plan = dict(action='auto', message='处理', options={'auto_train': True})
            with patch('root_self_train.__main__.load', return_value=cfg), \
                    patch('root_self_train.agent.request_plan', return_value=plan), \
                    patch('root_self_train.agent.execute') as executor, patch('builtins.print') as output:
                result = ask('config.yaml', '处理图片')
            self.assertEqual(result['status'], 'needs_setup')
            self.assertEqual(len(result['missing_files']), 2)
            self.assertIn('只分割', result['message'])
            self.assertNotIn('Errno', str(output.call_args_list))
            executor.assert_not_called()
            self.assertFalse(Path(cfg['workdir']).exists())
            self.assertIsNone(missing_data_guidance('predict', cfg))
            Path(cfg['val_manifest']).write_text('{}', encoding='utf-8')
            self.assertIsNone(missing_data_guidance('evaluate', cfg))
            self.assertEqual(len(missing_data_guidance('run', cfg)['missing_files']), 1)

    def test_chat_keeps_running_after_error_and_clear(self):
        seen = []
        def turn(config, instruction, dry_run=False, history=None):
            seen.append((instruction, len(history)))
            history.append({'role': 'user', 'content': instruction})
            if instruction == '失败':
                raise RuntimeError('模拟失败')
            return {'action': 'reply', 'message': '你好', 'options': {}}
        with patch('root_self_train.__main__.load', return_value=self.cfg), \
                patch('root_self_train.agent.ask', side_effect=turn), \
                patch('builtins.input', side_effect=['', '你好', '失败', '/result', '/clear', '继续', 'exit']), \
                patch('builtins.print') as output:
            chat('config.yaml')
        self.assertEqual(seen, [('你好', 0), ('失败', 1), ('继续', 0)])
        self.assertIn('模拟失败', str(output.call_args_list))

    def test_followup_uses_user_path_without_sending_local_results(self):
        history = []
        plans = [dict(action='clarify', message='需要执行什么任务？', options={}),
                 dict(action='predict', message='分割', options={'input_dir': 'my data'})]
        with patch('root_self_train.__main__.load', side_effect=lambda _: self.cfg.copy()), \
                patch('root_self_train.agent.request_plan', side_effect=plans) as planner, \
                patch('root_self_train.agent.execute') as executor, patch('builtins.print'):
            ask('config.yaml', '图片在 my data', history=history)
            result = ask('config.yaml', '分割它们', dry_run=True, history=history)
        self.assertEqual(len(planner.call_args.kwargs['history']), 2)
        self.assertEqual(len(history), 4)
        self.assertEqual(result['options']['input_dir'], 'my data')
        self.assertNotIn(result['output'], json.dumps(history))
        executor.assert_not_called()

    def test_provider_receives_dialogue_history(self):
        history = [dict(role='user', content='你好'), dict(role='assistant', content='你好！')]
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(
            dict(action='reply', message='可以分割根系', options={}))}}]}
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test'}, clear=True), \
                patch('root_self_train.agent.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as call:
            request_plan('能做什么', self.cfg, history=history)
        self.assertEqual(json.loads(call.call_args.args[0].data)['messages'][1:-1], history)

    def test_history_is_bounded_and_reply_does_not_execute(self):
        history = []
        plan = dict(action='reply', message='你好！', options={})
        with patch('root_self_train.__main__.load', return_value=self.cfg), \
                patch('root_self_train.agent.request_plan', return_value=plan), \
                patch('root_self_train.agent.execute') as executor, patch('builtins.print'):
            for i in range(15):
                ask('config.yaml', f'你好{i}', history=history)
        self.assertEqual(len(history), 24)
        self.assertEqual(history[0]['content'], '你好3')
        executor.assert_not_called()

    def test_cli_without_instruction_enters_chat(self):
        from root_self_train.__main__ import main
        with patch('sys.argv', ['root_self_train', 'ask', 'config.yaml']), \
                patch('root_self_train.agent.chat') as session:
            main()
        session.assert_called_once_with('config.yaml', dry_run=False)

    def test_eof_and_interrupt_exit_cleanly(self):
        for exception in (EOFError, KeyboardInterrupt):
            with patch('root_self_train.__main__.load', return_value=self.cfg), \
                    patch('builtins.input', side_effect=exception), patch('builtins.print'):
                chat('config.yaml')

    def test_dry_run_cannot_silently_start_legacy_training(self):
        from root_self_train.__main__ import main
        with patch('sys.argv', ['root_self_train', 'run', 'missing.yaml', '--dry-run']), \
                patch('sys.stderr', new_callable=io.StringIO), \
                patch('root_self_train.__main__.load') as loader:
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            loader.assert_not_called()

    def setUp(self):
        self.cfg = dict(rounds=3, iters=1000, auto_train_dice=.85,
                        unlabeled_dir='default', workdir='runs/test')

    def test_qwen_request_contract(self):
        plan = dict(action='predict', message='分割', options={})
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(plan)}}]}
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-secret'}, clear=True), \
                patch('root_self_train.agent.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as call:
            self.assertEqual(request_plan('分割这批图片', self.cfg), plan)
        request = call.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body['model'], 'qwen-plus')
        self.assertEqual(body['response_format'], {'type': 'json_object'})
        self.assertFalse(body['enable_thinking'])
        self.assertNotIn('test-secret', request.data.decode())
        self.assertNotIn('unlabeled_dir', request.data.decode())

    def test_errors_do_not_leak_provider_body(self):
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-secret'}, clear=True), \
                patch('root_self_train.agent.urlopen', side_effect=HTTPError('url', 401, 'secret', {}, None)):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 401') as error:
                request_plan('分割', self.cfg)
            self.assertNotIn('secret', str(error.exception))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'API Key'):
                request_plan('分割', self.cfg)

    def test_invalid_response_and_truncation(self):
        for response in ({}, {'choices': []}, {'choices': [{'finish_reason': 'length'}]},
                         {'choices': [{'finish_reason': 'stop', 'message': {'content': 'not json'}}]}):
            with self.subTest(response=response), patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test'}, clear=True), \
                    patch('root_self_train.agent.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
                with self.assertRaises(ValueError):
                    request_plan('分割', self.cfg)

    def test_reject_unbounded_or_unknown_operations(self):
        for action, options in [('shell', {}), ('predict', {'command': 'rm'}),
                                ('run', {'rounds': 4}), ('run', {'iters': True}),
                                ('auto', {}), ('auto', {'auto_train': 'false'}),
                                ('predict', {'input_dir': 'invented'}),
                                ('auto', {'auto_train': True, 'auto_train_dice': float('nan')}),
                                ('evaluate', {'input_dir': 'data'}), ('clarify', {'output': 'data'})]:
            with self.subTest(action=action, options=options), self.assertRaises(ValueError):
                validate_plan(dict(action=action, message='test', options=options), 'data', self.cfg)
        validate_plan(dict(action='auto', message='仅分割并评估', options={'auto_train': False}), '评估', self.cfg)

    def test_dry_run_and_clarification_do_not_execute(self):
        for action in ('predict', 'clarify'):
            with patch('root_self_train.__main__.load', return_value=self.cfg.copy()), \
                    patch('root_self_train.agent.request_plan', return_value=dict(action=action, message='说明', options={})), \
                    patch('root_self_train.agent.execute') as executor, patch('builtins.print'):
                ask('config.yaml', '分割', dry_run=action == 'predict')
                executor.assert_not_called()

    def test_new_input_clears_old_manifest_and_saves_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(self.cfg, unlabeled_manifest='old.jsonl')
            plan = dict(action='predict', message='分割', options={'input_dir': 'new data', 'output': 'output'})
            def fake_execute(plan, effective):
                self.assertNotIn('unlabeled_manifest', effective)
                self.assertEqual(effective['unlabeled_dir'], str(root / 'new data'))
                Path(effective['workdir']).mkdir()
                return {'status': 'completed'}
            with patch('root_self_train.__main__.load', return_value=cfg), \
                    patch('root_self_train.agent.request_plan', return_value=plan), \
                    patch('root_self_train.agent.execute', side_effect=fake_execute), patch('builtins.print'):
                ask(root / 'config.yaml', '分割 new data 输出 output')
            self.assertTrue((root / 'output/agent_task.json').is_file())

    def test_predict_uses_manifest_and_returns_actual_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'images.jsonl'
            manifest.write_text(json.dumps({'image': 'one.png'}), encoding='utf-8')
            cfg = dict(self.cfg, model_config='model.yml', checkpoint='model.pdparams', device='cpu',
                       workdir=str(root / 'out'), unlabeled_manifest=str(manifest))
            with patch('root_self_train.backend.Predictor'), \
                    patch('root_self_train.__main__.predict_directory',
                          side_effect=lambda p, images, out: (out.mkdir(), [{}])[1]) as predict:
                result = execute({'action': 'predict'}, cfg)
            self.assertEqual(result['images'], 1)
            self.assertEqual(predict.call_args.args[1], [root / 'one.png'])

    def test_auto_does_not_report_below_target_as_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'auto_summary.json').write_text(json.dumps(dict(final={'dice': .7}, checkpoint='old')), encoding='utf-8')
            with patch('root_self_train.__main__.auto') as auto:
                result = execute({'action': 'auto'}, dict(self.cfg, workdir=tmp, auto_train=False))
            self.assertFalse(result['validation_target_met'])
            self.assertFalse(auto.call_args.args[0]['auto_train'])


if __name__ == '__main__':
    unittest.main()
