"""Read-only PyTorch device inspection and official wheel selection."""
import json
import platform
import struct
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def probe(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, errors='replace', check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def inspect_environment():
    # 只读采集 Python、系统、已安装包和 NVIDIA 驱动信息，不安装驱动或修改设备设置。
    packages = {}
    for name in ('pip', 'numpy', 'torch'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=list(sys.version_info[:3]), executable=sys.executable,
                system=platform.system(), machine=platform.machine(), bits=struct.calcsize('P')*8,
                packages=packages, gpu_report=probe(['nvidia-smi', '--query-gpu=index,name,driver_version', '--format=csv,noheader']))


def torch_plan(env, device):
    # 返回安装建议而非直接执行；存在 blockers 时 command 留空，避免盲目重复安装。
    issues = []
    gpu = str(device).startswith(('gpu', 'cuda'))
    if env['system'] not in {'Windows', 'Linux'} or env['bits'] != 64 or env['machine'].lower() not in {'amd64', 'x86_64', 'x64'}:
        issues.append('自动安装仅覆盖 Windows/Linux x64；其他平台请按 PyTorch 官方说明安装。')
    if not (3, 10) <= tuple(env['python'][:2]) <= (3, 13):
        issues.append('固定 PyTorch 2.8 自动安装范围为 Python 3.10–3.13；Deep Agents 请使用 3.11–3.13。')
    if env['packages'].get('torch'):
        issues.append('已安装 torch 但导入失败，请修复现有环境，不自动覆盖。')
    if gpu and not env['gpu_report']:
        issues.append('无法检测 NVIDIA 驱动；请先安装驱动，或显式设置 device: cpu。')
    if not gpu and device != 'cpu':
        issues.append('设备应为 cpu 或 gpu/cuda。')
    index = 'cu128' if gpu else 'cpu'
    # 这里选择 PyTorch 自带运行时的 wheel 源，不代表本机已经满足 GPU 运行条件。
    return dict(environment=env, device=device, blockers=issues,
                command=[] if issues else [sys.executable, '-m', 'pip', 'install', 'torch==2.8.0', '--index-url', 'https://download.pytorch.org/whl/' + index],
                reason='官方 CPU/CUDA 12.8 wheel；安装后仍需检测设备是否可用，不安装驱动或 CUDA Toolkit。')


def reviewed_torch_command(device):
    plan = torch_plan(inspect_environment(), device)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if plan['blockers']:
        raise RuntimeError('环境检查未通过：\n' + '\n'.join(plan['blockers']))
    return plan['command']


if __name__ == '__main__':
    print(json.dumps(torch_plan(inspect_environment(), sys.argv[1] if len(sys.argv)>1 else 'gpu'), ensure_ascii=False, indent=2))
