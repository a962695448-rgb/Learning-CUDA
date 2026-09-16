"""Build exact old/new sources, validate out buffers, and time ordinary calls."""
from pathlib import Path
import functools
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/os.environ.get('API_OPT_RESULTS_DIR','results');OUT.mkdir()
os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH']
os.environ['TORCH_CUDA_ARCH_LIST']='8.9';os.environ['MAX_JOBS']='1'
manifest=json.loads((ROOT/'INPUT_MANIFEST.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
import numpy as np
import torch
import triton
from torch.utils.cpp_extension import load
project=Path('03_hadamard_tc/a962695448-rgb')
sys.path.insert(0,str(ROOT/'candidate/cuda'/project/'scripts'))
from verify_out_buffers import buffers_for,positive_cases,negative_cases,mutation_contract,execution_contexts
from verify_execution_context import compare
protocol=json.loads((ROOT/'PROTOCOL.json').read_text())['cuda']
summary={'status':'RUNNING','jobs':[],'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
         'triton':triton.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'arch_list':os.environ['TORCH_CUDA_ARCH_LIST'],
         'capability':list(torch.cuda.get_device_capability()),'nvcc':subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True),
         'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)},
         'input_manifest_sha256':hashlib.sha256((ROOT/'INPUT_MANIFEST.json').read_bytes()).hexdigest(),
         'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest()}


def save(name,data): (OUT/name).write_text(json.dumps(data,indent=2)+'\n')


def module(version):
    root=ROOT/version/'cuda'/project;build=ROOT/'build'/version;build.mkdir(parents=True,exist_ok=True)
    flags=['-O3','-std=c++17','-lineinfo','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
           '-U__CUDA_NO_BFLOAT16_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','--expt-relaxed-constexpr']
    return load(name='hadamard_out_'+version,sources=[str(root/'src/torch_binding.cu')],
                extra_include_paths=[str(root/'include')],extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=flags,
                build_directory=str(build),with_cuda=True,verbose=True)


def call(op,x,method,buffers=None):
    if method=='hadamard':
        return functools.partial(op.hadamard,x,1.,128,'original') if buffers is None else functools.partial(op.hadamard_out,x,*buffers,1.,128,'original')
    if method=='hadamard_int4':
        return functools.partial(op.hadamard_int4,x,1.,128,'original','original') if buffers is None else functools.partial(op.hadamard_int4_out,x,*buffers,1.,128,'original','original')
    return functools.partial(op.quantize_int4,x,128) if buffers is None else functools.partial(op.quantize_int4_out,x,*buffers,128)


def timing(fn):
    torch.cuda.synchronize();start=time.perf_counter_ns()
    for _ in range(protocol['calls_per_group']):fn()
    torch.cuda.synchronize()
    return (time.perf_counter_ns()-start)/protocol['calls_per_group']/1000


save('summary.json',summary)
try:
    control,candidate=module('control'),module('candidate')
    summary['extension_sha256']={v:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for v,m in [('control',control),('candidate',candidate)]}
    correctness={'status':'RUNNING'};save('out-correctness.json',correctness)
    for key,fn,args in [('positive',positive_cases,(torch,np,candidate)),('negative',negative_cases,(torch,candidate)),
                        ('mutation',mutation_contract,(torch,np,candidate)),('contexts',execution_contexts,(torch,np,candidate))]:
        correctness[key]=fn(*args);save('out-correctness.json',correctness);print(key,len(correctness[key]),'PASS',flush=True)
    correctness['status']='PASS';save('out-correctness.json',correctness)
    report={'status':'RUNNING','protocol':protocol,'records':[],'rounds':[]};save('out-benchmark.json',report)
    for round_id in range(protocol['rounds']):
        rng=torch.Generator().manual_seed(93712026+round_id)
        for dtype in (torch.float16,torch.bfloat16):
            for rows in protocol['target_rows']+protocol['control_rows']:
                for dim in protocol['target_dims']:
                    x=torch.randn((rows,dim),generator=rng).to(dtype).cuda()
                    for method in protocol['methods']:
                        buffers,_=buffers_for(torch,x,method)
                        fns={'control':call(control,x,method),'current':call(candidate,x,method),'out':call(candidate,x,method,buffers)}
                        expected=fns['control']();expected=expected if isinstance(expected,tuple) else (expected,)
                        expected=tuple(v.cpu() for v in expected)
                        compare(torch,fns['current'](),expected);fns['out']();compare(torch,buffers,expected)
                        for fn in fns.values():
                            for _ in range(100):fn()
                        samples={name:[] for name in fns}
                        for group in range(protocol['groups']):
                            order=['control','current','out'];shift=(group+round_id)%3;order=order[shift:]+order[:shift]
                            for name in order:samples[name].append(timing(fns[name]))
                        out_ratio=statistics.median(samples['control'])/statistics.median(samples['out'])
                        legacy_ratio=statistics.median(samples['control'])/statistics.median(samples['current'])
                        report['records'].append({'round':round_id,'dtype':str(dtype),'rows':rows,'dim':dim,'method':method,
                          'target':rows in protocol['target_rows'],'host_us':samples,'out_speedup':out_ratio,
                          'current_speedup':legacy_ratio,'gate_passed':min(out_ratio,legacy_ratio)>=1/1.05})
                save('out-benchmark.json',report)
        for method in protocol['methods']:
            records=[r for r in report['records'] if r['round']==round_id and r['method']==method and r['target']]
            ratio=math.exp(statistics.mean(math.log(r['out_speedup']) for r in records))
            report['rounds'].append({'round':round_id,'method':method,'geomean_out_speedup':ratio,'gate_passed':ratio>=1.10})
        save('out-benchmark.json',report);print('round',round_id,'finished',flush=True)
    report['status']='ACCEPT' if all(r['gate_passed'] for r in report['records']+report['rounds']) else 'REJECT';save('out-benchmark.json',report)
    summary['cuda_decision']=report['status']
except Exception:
    summary['cuda_error']=traceback.format_exc();print(summary['cuda_error'],flush=True)

# Test the CPU interpreter candidate against actual GPU code after timing finishes.
nine=ROOT/'candidate/nine';env=dict(os.environ,PYTHONPATH=str(nine/'src'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
jobs=[] if os.environ.get('API_OPT_CUDA_ONLY')=='1' else [('nine-gpu',['scripts/verify_interpreter_gpu.py','--report',str(OUT/'nine-gpu.json')]),
                  ('nine-dtype-tests',['-m','pytest','-q','tests/test_interpreter_dtype_cache.py','--junitxml='+str(OUT/'nine-dtype-tests.xml')])]
for name,args in jobs:
    command=[sys.executable,*args];start=time.monotonic()
    with (OUT/f'{name}.log').open('x') as log:
        result=subprocess.run(command,cwd=nine,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    summary['jobs'].append({'name':name,'returncode':result.returncode,'seconds':time.monotonic()-start,'command':command})
for name,sha in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
summary['sources_unchanged_during_run']=True
summary['status']='PASS' if 'cuda_error' not in summary and all(j['returncode']==0 for j in summary['jobs']) else 'FAIL';save('summary.json',summary)
files={f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'text':f.read_bytes().decode()} for f in OUT.iterdir() if f.is_file()}
payload=json.dumps(files,ensure_ascii=True);(OUT/'API_EXPORT.txt').write_text(payload)
print(json.dumps({'status':summary['status'],'cuda_decision':summary.get('cuda_decision'),'bytes':len(payload.encode()),'sha256':hashlib.sha256(payload.encode()).hexdigest()}),flush=True)
