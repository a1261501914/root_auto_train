import unittest
from root_self_train.environment import torch_plan

class EnvironmentTests(unittest.TestCase):
    def test_cpu_gpu_and_unsupported_python(self):
        env = dict(system='Windows', bits=64, machine='AMD64', python=[3,13,5], packages={}, gpu_report='0, RTX 5070, 580')
        self.assertTrue(torch_plan(env,'cpu')['command'][-1].endswith('/cpu'))
        self.assertTrue(torch_plan(env,'gpu')['command'][-1].endswith('/cu128'))
        self.assertTrue(torch_plan(dict(env, python=[3,9,0]), 'cpu')['blockers'])
        self.assertTrue(torch_plan(dict(env, gpu_report=''), 'gpu')['blockers'])
        self.assertTrue(torch_plan(dict(env, packages={'torch':'2.8.0'}), 'cpu')['blockers'])
