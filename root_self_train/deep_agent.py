"""Deep Agents supervisor and Qwen specialists for validated task delegation.

Tools submit plans only. Local execution happens after the graph completes, so
dry-run cannot train and local results never enter remote model context.
"""
import json
import os
from threading import Lock
from urllib.parse import urlparse


class PlanValidationError(ValueError):
    """Locally generated validation feedback, safe to display without provider bodies."""


def planning_error(exc):
    name = type(exc).__name__
    status = getattr(exc, 'status_code', None)
    if isinstance(exc, PlanValidationError):
        detail = '任务计划校验失败：' + str(exc)
    elif name == 'GraphRecursionError':
        detail = '智能体委派超过步骤上限，未能结束规划；请重试或用 /clear 清空上下文'
    elif name in {'APIConnectionError', 'OpenAIConnectionError', 'ConnectError', 'ConnectionError'}:
        detail = '无法连接千问接口；请检查网络、代理和 qwen.base_url'
    elif name in {'APITimeoutError', 'OpenAITimeoutError', 'TimeoutError', 'ReadTimeout'}:
        detail = '千问接口请求超时；请稍后重试或调整 qwen.timeout'
    elif status in (401, 403):
        detail = f'千问接口拒绝访问（HTTP {status}）；请检查 API Key、地域和模型权限'
    elif status == 429:
        detail = '千问接口限流或额度受限（HTTP 429）；请检查调用频率和账户额度'
    elif status in (400, 404, 422):
        detail = f'千问接口拒绝请求（HTTP {status}）；请检查模型名称和工具调用参数兼容性'
    else:
        detail = f'规划出现 {name}' + (f'（HTTP {status}）' if isinstance(status, int) else '')
    return 'Deep Agents 规划未完成；没有执行本地任务。' + detail


class PlanInbox:
    # 计划收件箱：子智能体的工具只登记计划，真正的本地任务要等规划结束后才执行。
    def __init__(self, instruction, cfg, history):
        self.instruction, self.cfg = instruction, cfg
        self.evidence = '\n'.join([m['content'] for m in history if m['role'] == 'user'] + [instruction])
        self.plan = None
        self.lock = Lock()

    def submit(self, plan, allowed):
        from .agent import validate_plan
        try:
            plan = validate_plan(plan, self.evidence, self.cfg)
        except ValueError as exc:
            raise PlanValidationError(str(exc)) from None
        if plan['action'] not in allowed:
            raise PlanValidationError(f'当前工具只允许 {", ".join(sorted(allowed))}，收到 {plan["action"]}')
        if 'output' in plan['options'] and plan['options']['output'] not in self.instruction:
            raise PlanValidationError('输出目录须在本轮明确提供')
        with self.lock:
            # 多个子智能体可能提交结果；锁保证本轮不会登记两个相互冲突的执行计划。
            if self.plan is not None and self.plan != plan:
                raise PlanValidationError('本轮已提交任务，不能提交冲突任务；复合流程使用 auto')
            self.plan = plan
        return '任务计划已登记，尚未执行。执行结果只在本地显示。'


def build_graph(model, inbox):
    # 构建主控和三类规划子智能体；它们与 specialists.py 的本地执行对象职责不同。
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from langchain_core.tools import tool
    from .agent import SYSTEM

    def make_tool(name, actions):
        # 闭包把各角色允许的 action 固定下来，例如分割角色只能提交 predict。
        @tool(name)
        def submit_task(action: str, message: str, options: dict) -> str:
            """Submit one structured task plan. This does not execute or evaluate a model."""
            try:
                return inbox.submit(dict(action=action, message=message, options=options), actions)
            except PlanValidationError as exc:
                # 可纠正的工具参数错误反馈给规划模型，不执行无效计划。
                if inbox.plan is not None:
                    return '提交未接受：' + str(exc) + '。已有有效计划，请结束本轮，不要再次提交。'
                return '提交未接受：' + str(exc) + '。请使用正确角色和参数重试，执行任务须通过 task 委派。'
        submit_task.description += ' Allowed action values: ' + ', '.join(sorted(actions)) + '. Invalid plans are rejected without execution.'
        return submit_task

    roles = [('segmentation', '根系分割任务规划，只负责 predict。', {'predict'}),
             ('evaluation', '新批次无标签质量筛查 assess、人工验证集 evaluate 和数据检查。', {'assess', 'evaluate', 'check'}),
             ('training', '有预算的自训练、视觉低分触发适配 adapt 或复合流程规划。', {'run', 'auto', 'adapt'})]
    budget = json.dumps({k: inbox.cfg[k] for k in ('rounds', 'iters', 'auto_train_dice')})
    # JSON-only 是单模型解析器约束，此处须调用工具，不能同时要求仅输出 JSON。
    task_rules = SYSTEM.replace('只输出 JSON，不执行任务，不宣称任务完成。', '通过工具提交任务，不执行任务，不宣称任务完成。').replace(
        '严格输出 {"action":字符串,"message":中文任务说明或澄清问题,"options":对象}。',
        '提交工具的参数为 action、message（中文说明）、options（对象）。')
    prompt = task_rules + '\n配置预算：' + budget + '''
这是 Deep Agents 调度会话。用提交工具登记任务，不能仅返回任务 JSON。
只说处理图片、未要求评估或训练时，默认仅分割。新批次分割并质检由 evaluation 提交 assess；需要训练的复合流程由 training 提交 auto，
不要同时提交多个单项任务。所有提交仅代表计划，不代表执行成功。
'''
    subagents = [dict(name=name, description=description, model=model,
                      system_prompt=prompt + '\n' + description + '\n调用你的 submit 工具提交一个计划。',
                      tools=[make_tool('submit_' + name, actions)])
                 for name, description, actions in roles]
    return create_deep_agent(
        # StateBackend 使用内存状态，不把本地目录或 shell 作为规划模型的工具。
        model=model, backend=StateBackend(),
        system_prompt=prompt + '''
你是主控。执行类请求必须通过 task 委派 segmentation/evaluation/training，
委派时提供用户原始指令以及所需历史用户信息。问候或澄清使用 respond 工具，
action 为 reply 或 clarify，options={}。不要自行读取本地文件或运行 shell。
respond 不能提交 predict/assess/evaluate/check/auto/run。子智能体登记计划后直接结束本轮，不再用 respond 登记总结。
''', tools=[make_tool('respond', {'reply', 'clarify'})], subagents=subagents)


def request_deep_plan(instruction, cfg, history=None):
    # 按需导入，直接使用本地分割命令时无需先加载整个智能体框架。
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
            # 关闭追踪；recursion_limit 限制委派调用深度，避免规划无限循环。
            graph.invoke({'messages': history + [{'role': 'user', 'content': instruction}]},
                         config={'recursion_limit': 32})
    except Exception as exc:
        raise RuntimeError(planning_error(exc)) from None
    if inbox.plan is None:
        raise RuntimeError('Deep Agents 未提交有效任务；没有执行。请补充具体要求后重试。')
    return inbox.plan
