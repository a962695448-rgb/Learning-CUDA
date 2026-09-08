"""Verify a real CUDA launch and Nsight counter access; do not change driver policy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory.')
    parser.add_argument('--arch', choices=('80', '89'), required=True)
    parser.add_argument('--sudo-profile', action='store_true', help='Use existing noninteractive sudo for ncu only.')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'expected_sm': args.arch, 'sudo_profile': args.sudo_profile, 'commands': [],
              'notes': ['This probes profiling capability, not project performance or exclusive tenancy.',
                        'No module parameters, persistence, power limits, or clock locks are changed.']}
    def run(label, command, required=True):
        command = list(map(str, command))
        record = {'label': label, 'argv': command}
        report['commands'].append(record)
        with (args.output / (label + '.stdout.log')).open('xb') as stdout, (args.output / (label + '.stderr.log')).open('xb') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=120)
        record['exit_code'] = result.returncode
        if required and result.returncode:
            raise RuntimeError(label + ' failed; see retained logs')
        return (args.output / (label + '.stdout.log')).read_text(errors='replace')
    code = 1
    try:
        if sys.platform != 'linux':
            raise RuntimeError('Run this probe on the Linux GPU server.')
        nvcc, ncu = shutil.which('nvcc'), shutil.which('ncu')
        if not nvcc or not ncu:
            raise RuntimeError('nvcc and ncu must be installed and available on PATH.')
        run('device', ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,mig.mode.current,utilization.gpu', '--format=csv'])
        processes = run('compute-processes', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'])
        if processes.strip():
            raise RuntimeError('Visible compute processes exist; obtain an idle profiling window first.')
        run('telemetry', ['nvidia-smi', '--query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit', '--format=csv'])
        run('nvcc-version', [nvcc, '--version'])
        run('ncu-version', [ncu, '--version'])
        source = Path(__file__).with_name('hardware_probe.cu').resolve()
        report['probe_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        binary = args.output / 'hardware_probe'
        run('compile', [nvcc, '-O2', '-lineinfo', '-arch=sm_' + args.arch, source, '-o', binary])
        output = run('cuda-launch', [binary])
        if 'PROBE_CORRECTNESS_PASS' not in output or 'SM=' + args.arch + ' ' not in output:
            raise RuntimeError('GPU architecture or actual launch correctness differs from the request.')
        prefix = ['sudo', '-n'] if args.sudo_profile else []
        run('ncu', [*prefix, ncu, '--set', 'basic', '--launch-count', '1', '--clock-control', 'none',
                    '--csv', '--log-file', args.output / 'ncu-metrics.csv', binary])
        metrics = (args.output / 'ncu-metrics.csv').read_text(errors='replace')
        launch = (args.output / 'ncu.stdout.log').read_text(errors='replace')
        if 'ERR_NVGPUCTRPERM' in metrics or '==ERROR==' in metrics:
            raise RuntimeError('Nsight reported a profiling error.')
        if 'PROBE_CORRECTNESS_PASS' not in launch or 'profiling_capability_probe' not in metrics or 'Metric Name' not in metrics:
            raise RuntimeError('A successful ncu exit without actual kernel metrics is insufficient.')
        report['status'] = 'PASS'
        code = 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        report.update(status='FAIL', error=str(error))
    finally:
        report.update(exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat())
        report['artifacts'] = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                               for p in args.output.iterdir() if p.is_file()}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'report': str(args.output / 'report.json'), 'error': report.get('error')}))
    return code

if __name__ == '__main__':
    raise SystemExit(main())
