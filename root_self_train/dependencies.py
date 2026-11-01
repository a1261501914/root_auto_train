"""Install known dependencies only after explicit interactive consent."""
import importlib
from importlib import import_module
import subprocess
import sys


PACKAGES = {
    'numpy': 'numpy>=1.24,<3',
    'PIL': 'Pillow>=10',
    'yaml': 'PyYAML>=6',
    'torch': 'torch==2.8.0',
    'cv2': 'opencv-python',
    'deepagents': 'deepagents==0.7.32',
    'langchain_openai': 'langchain-openai==1.7.3',
}


def ensure_imports(modules, device='cpu'):
    """Retry imports only; never replay a partially completed pipeline.

    Unknown modules and non-missing-module import failures are left untouched.
    Each distribution is attempted at most once per call.
    """
    modules = tuple(modules)
    attempted = set()
    packages = PACKAGES
    for module in modules:
        while True:
            try:
                import_module(module)
                break
            except ModuleNotFoundError as exc:
                name = exc.name
                package = packages.get(name)
                if package is None:
                    raise
                if package in attempted:
                    raise RuntimeError(f'{package} 安装后仍无法导入 {name}，请检查环境兼容性并重启程序') from exc
                if name == 'deepagents' and not (3, 11) <= tuple(sys.version_info[:2]) < (4, 0):
                    raise RuntimeError(
                        'Deep Agents 0.7.32 要求 Python >=3.11 且 <4.0；'
                        f'当前为 {sys.version_info[0]}.{sys.version_info[1]}（{sys.executable}）。\n'
                        '已停止安装，重试 pip 无法解决此版本冲突。\n'
                        '请新建 Python 3.11–3.13 环境运行 PyTorch + Deep Agents；也可临时设置 agent_backend: legacy。'
                    ) from None
                command = [sys.executable, '-m', 'pip', 'install', package]
                if name == 'torch':
                    from .environment import reviewed_torch_command
                    command = reviewed_torch_command(device)
                print(f'缺少依赖：{name}。将安装到当前 Python：{sys.executable}', flush=True)
                print('安装命令：' + subprocess.list2cmdline(command), flush=True)
                if name == 'torch':
                    print(f'当前设备配置：{device}。GPU 版需要匹配的驱动/CUDA；安装成功不代表设备兼容。')
                print('此操作会联网下载依赖及其依赖包，并修改上述 Python 环境。', flush=True)
                try:
                    answer = input('是否同意安装并继续？[y/N]：').strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = ''
                if answer not in {'y', 'yes', '是', '同意'}:
                    missing = {name}
                    for remaining in modules:
                        if remaining == module:
                            continue
                        try:
                            import_module(remaining)
                        except ModuleNotFoundError as other:
                            missing.add(other.name or remaining)
                        except ImportError:
                            missing.add(remaining + '（导入失败）')
                    details = '\n'.join(f'- {item}' + (f'：{packages[item]}' if item in packages else '')
                                        for item in sorted(missing))
                    raise RuntimeError(
                        '未获得安装同意，本次没有执行此安装，任务尚未继续。\n'
                        '当前步骤检测到缺少或无法导入的依赖：\n' + details + '\n'
                        '如需手动安装当前依赖，可运行：\n' + subprocess.list2cmdline(command) + '\n'
                        '安装完成后请重试任务；其他未执行步骤可能还有额外依赖。'
                    ) from None
                print('已获得同意，正在安装，完成后继续当前任务……', flush=True)
                attempted.add(package)
                try:
                    subprocess.run(command, check=True, timeout=600)
                except (subprocess.SubprocessError, OSError) as error:
                    raise RuntimeError(f'{package} 安装失败，请查看 pip 输出并检查网络、Python/设备兼容性后重试') from error
                importlib.invalidate_caches()
