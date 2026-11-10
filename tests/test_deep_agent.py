import importlib.util
import unittest

from root_self_train.deep_agent import PlanInbox, build_graph, planning_error


class InboxTests(unittest.TestCase):
    def test_provider_error_is_sanitized(self):
        class OpenAIConnectionError(Exception):
            pass
        detail = planning_error(OpenAIConnectionError('secret-key / private response body'))
        self.assertIn('无法连接', detail)
        self.assertNotIn('secret-key', detail)

    def test_role_budget_and_duplicate_limits(self):
        inbox = PlanInbox('分割 data', dict(rounds=2, iters=10), [])
        plan = dict(action='predict', message='分割', options={'input_dir': 'data'})
        inbox.submit(plan, {'predict'})
        inbox.submit(plan, {'predict'})
        with self.assertRaises(ValueError):
            inbox.submit(dict(action='run', message='训练', options={}), {'predict'})
        with self.assertRaises(ValueError):
            inbox.submit(dict(action='run', message='训练', options={'iters': 20}), {'run'})
        with self.assertRaises(ValueError):
            inbox.submit(dict(action='evaluate', message='评估', options={}), {'evaluate'})


@unittest.skipUnless(importlib.util.find_spec('deepagents'), 'optional Deep Agents dependencies')
class GraphTests(unittest.TestCase):
    def test_wrong_role_tool_call_can_recover(self):
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage, ToolMessage
        from langsmith import tracing_context

        class Model(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        model = Model(responses=[
            AIMessage(content='', tool_calls=[dict(name='respond', args=dict(
                action='assess', message='评估', options={'input_dir':'data'}), id='wrong')]),
            AIMessage(content='', tool_calls=[dict(name='task', args=dict(
                subagent_type='evaluation', description='分割并评估 data，不训练'), id='delegate')]),
            AIMessage(content='', tool_calls=[dict(name='submit_evaluation', args=dict(
                action='assess', message='评估', options={'input_dir':'data'}), id='submit')]),
            AIMessage(content='已登记。'),
            AIMessage(content='计划已登记。'),
        ])
        inbox = PlanInbox('分割并评估 data', dict(rounds=2,iters=10,auto_train_dice=.85), [])
        with tracing_context(enabled=False):
            result=build_graph(model,inbox).invoke({'messages':[{'role':'user','content':'分割并评估 data'}]},
                                                 config={'recursion_limit':16})
        self.assertEqual(inbox.plan['action'],'assess')
        self.assertTrue(any(isinstance(m,ToolMessage) and '提交未接受' in str(m.content) for m in result['messages']))

    def test_real_graph_delegates_to_segmentation(self):
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage
        from langsmith import tracing_context

        class Model(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        model = Model(responses=[
            AIMessage(content='', tool_calls=[dict(name='task', args=dict(
                subagent_type='segmentation', description='只分割 data'), id='delegate')]),
            AIMessage(content='', tool_calls=[dict(name='submit_segmentation', args=dict(
                action='predict', message='分割', options={'input_dir': 'data'}), id='submit')]),
            AIMessage(content='计划已提交，尚未执行。'),
            AIMessage(content='计划已提交。'),
        ])
        inbox = PlanInbox('只分割 data', dict(rounds=2, iters=10, auto_train_dice=.85), [])
        with tracing_context(enabled=False):
            graph = build_graph(model, inbox)
            graph.invoke({'messages': [{'role': 'user', 'content': '只分割 data'}]},
                         config={'recursion_limit': 16})
        self.assertEqual(inbox.plan['action'], 'predict')
