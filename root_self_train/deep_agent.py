"""Deep Agents supervisor and Qwen specialists for validated task delegation.

Tools submit plans only. Local execution happens after the graph completes, so
dry-run cannot train and local results never enter remote model context.
"""
import json
import os
from threading import Lock
from urllib.parse import urlparse


class PlanInbox:
    def __init__(self, instruction, cfg, history):
        self.instruction, self.cfg = instruction, cfg
        self.evidence = '\n'.join([m['content'] for m in history if m['role'] == 'user'] + [instruction])
        self.plan = None
        self.lock = Lock()

    def submit(self, plan, allowed):
        from .agent import validate_plan
        plan = validate_plan(plan, self.evidence, self.cfg)
        if plan['action'] not in allowed:
            raise ValueError('该子智能体不能提交此任务类型')
        if 'output' in plan['options'] and plan['options']['output'] not in self.instruction:
            raise ValueError('输出目录须在本轮明确提供')
        with self.lock:
            if self.plan is not None and self.plan != plan:
                raise ValueError('本轮已提交任务，不能提交冲突任务；复合流程使用 auto')
            self.plan = plan
        return '任务计划已登记，尚未执行。执行结果只在本地显示。'


def build_graph(model, inbox):
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from langchain_core.tools import tool
    from .agent import SYSTEM

    def make_tool(name, actions):
        @tool(name)
        def submit_task(action: str, message: str, options: dict) -> str:
            """Submit one structured task plan. This does not execute or evaluate a model."""
            return inbox.submit(dict(action=action, message=message, options=options), actions)
        return submit_task

    roles = [('segmentation', '根系分割任务规划，只负责 predict。', {'predict'}),
             ('evaluation', '人工验证集评估和数据检查任务规划。', {'evaluate', 'check'}),
             ('training', '有预算的自训练或分割评估训练复合流程规划。', {'run', 'auto'})]
    budget = json.dumps({k: inbox.cfg[k] for k in ('rounds', 'iters', 'auto_train_dice')})
    prompt = SYSTEM + '\n配置预算：' + budget + '''
这是 Deep Agents 调度会话。用提交工具登记任务，不能仅返回任务 JSON。
只说处理图片、未要求评估或训练时，默认仅分割。复合流程由 training 提交 auto，
不要同时提交多个单项任务。所有提交仅代表计划，不代表执行成功。
'''
    subagents = [dict(name=name, description=description, model=model,
                      system_prompt=prompt + '\n' + description + '\n调用你的 submit 工具提交一个计划。',
                      tools=[make_tool('submit_' + name, actions)])
                 for name, description, actions in roles]
    return create_deep_agent(
        model=model, backend=StateBackend(),
        system_prompt=prompt + '''
你是主控。执行类请求必须通过 task 委派 segmentation/evaluation/training，
委派时提供用户原始指令以及所需历史用户信息。问候或澄清使用 respond 工具，
action 为 reply 或 clarify，options={}。不要自行读取本地文件或运行 shell。
''', tools=[make_tool('respond', {'reply', 'clarify'})], subagents=subagents)


def request_deep_plan(instruction, cfg, history=None):
    from .dependencies import ensure_imports
    ensure_imports(('deepagents', 'langchain_openai'))
    try:
        from langchain_openai import ChatOpenAI
        from langsmith import tracing_context
        import deepagents  # noqa: F401
    except ImportError:
        raise RuntimeError('缺少 Deep Agents 依赖，请运行：python -m pip install -r requirements-agent.txt') from None
    settings = cfg.get('qwen', {})
    key = os.environ.get(settings.get('api_key_env', 'DASHSCOPE_API_KEY'))
    if not key:
        raise ValueError('请设置 DASHSCOPE_API_KEY 环境变量')
    base = os.environ.get('QWEN_BASE_URL') or settings.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    parsed = urlparse(base)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('千问 base_url 必须为无凭据和查询参数的 HTTPS 地址')
    timeout = settings.get('timeout', 60)
    if type(timeout) not in (int, float) or not 0 < timeout <= 300:
        raise ValueError('qwen.timeout 必须在 0 到 300 秒之间')
    model = ChatOpenAI(model=os.environ.get('QWEN_MODEL') or settings.get('model', 'qwen-plus'),
                       api_key=key, base_url=base, temperature=0, timeout=timeout, max_retries=1,
                       extra_body={'enable_thinking': False})
    history = list(history or [])
    inbox = PlanInbox(instruction, cfg, history)
    graph = build_graph(model, inbox)
    try:
        with tracing_context(enabled=False):
            graph.invoke({'messages': history + [{'role': 'user', 'content': instruction}]},
                         config={'recursion_limit': 32})
    except Exception:
        raise RuntimeError('Deep Agents 规划未完成；没有执行本地任务。请检查千问模型的工具调用支持、连接及额度，或简化指令重试。') from None
    if inbox.plan is None:
        raise RuntimeError('Deep Agents 未提交有效任务；没有执行。请补充具体要求后重试。')
    return inbox.plan
