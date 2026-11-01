import subprocess
import sys
import unittest
from unittest.mock import patch

from root_self_train.dependencies import ensure_imports


class DependencyTests(unittest.TestCase):
    def test_old_python_stops_before_pip_with_clear_guidance(self):
        with patch('root_self_train.dependencies.import_module', side_effect=self.missing('deepagents')), \
                patch('root_self_train.dependencies.sys.version_info', (3, 10, 11)), \
                patch('subprocess.run') as installer:
            with self.assertRaisesRegex(RuntimeError, 'agent_backend: legacy'):
                ensure_imports(('deepagents',))
            installer.assert_not_called()

    def setUp(self):
        planner = patch('root_self_train.environment.reviewed_torch_command',
                        side_effect=lambda device: [sys.executable, '-m', 'pip', 'install',
                            'torch==2.8.0'])
        planner.start()
        self.addCleanup(planner.stop)

    def missing(self, name='torch'):
        return ModuleNotFoundError('missing', name=name)

    def test_consent_installs_gpu_in_current_interpreter_and_retries(self):
        with patch('root_self_train.dependencies.import_module', side_effect=[self.missing(), object()]) as imp, \
                patch('builtins.input', return_value='同意'), patch('subprocess.run') as run:
            ensure_imports(('torch',), 'gpu:0')
        run.assert_called_once_with([sys.executable, '-m', 'pip', 'install', 'torch==2.8.0'], check=True, timeout=600)
        self.assertEqual(imp.call_count, 2)

    def test_deepagents_installs_after_consent(self):
        with patch('root_self_train.dependencies.import_module', side_effect=[self.missing('deepagents'), object()]), \
                patch('root_self_train.dependencies.sys.version_info', (3, 11, 0)), \
                patch('builtins.input', return_value='yes'), patch('subprocess.run') as run:
            ensure_imports(('deepagents',))
        self.assertEqual(run.call_args.args[0][-1], 'deepagents==0.7.32')

    def test_install_timeout_is_actionable(self):
        with patch('root_self_train.dependencies.import_module', side_effect=self.missing('yaml')), \
                patch('builtins.input', return_value='y'), \
                patch('subprocess.run', side_effect=subprocess.TimeoutExpired('pip', 600)):
            with self.assertRaisesRegex(RuntimeError, '安装失败'):
                ensure_imports(('yaml',))

    def test_unknown_dependency_never_installs(self):
        with patch('root_self_train.dependencies.import_module', side_effect=self.missing('unknown')), patch('subprocess.run') as run:
            with self.assertRaises(ModuleNotFoundError):
                ensure_imports(('unknown',))
            run.assert_not_called()

    def test_refusal_empty_eof_and_interrupt_never_install(self):
        for reply in ('n', '', 'maybe', EOFError(), KeyboardInterrupt()):
            kwargs = {'side_effect': reply} if isinstance(reply, BaseException) else {'return_value': reply}
            with self.subTest(reply=reply), \
                    patch('root_self_train.dependencies.import_module', side_effect=lambda name: (_ for _ in ()).throw(self.missing(name))), \
                    patch('builtins.input', **kwargs), patch('subprocess.run') as installer:
                with self.assertRaises(RuntimeError) as error:
                    ensure_imports(('deepagents', 'langchain_openai'))
                self.assertIn('deepagents', str(error.exception))
                self.assertIn('langchain_openai', str(error.exception))
                self.assertIn('手动安装', str(error.exception))
                installer.assert_not_called()

    def test_install_failure_and_repeated_missing_stop(self):
        for failure in [None, subprocess.CalledProcessError(1, 'pip')]:
            with patch('root_self_train.dependencies.import_module', side_effect=self.missing()), \
                    patch('builtins.input', return_value='y'), patch('subprocess.run', side_effect=failure) as run:
                with self.assertRaises(RuntimeError):
                    ensure_imports(('torch',))
                self.assertEqual(run.call_count, 1)

    def test_present_dependency_does_not_prompt(self):
        with patch('root_self_train.dependencies.import_module'), patch('builtins.input') as prompt:
            ensure_imports(('numpy',))
            prompt.assert_not_called()

