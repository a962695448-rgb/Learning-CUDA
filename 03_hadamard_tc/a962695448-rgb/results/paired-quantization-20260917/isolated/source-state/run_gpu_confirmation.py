"""Validate paired-value quantization against the current packed implementation."""
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
OUT=ROOT/'results-confirmation';OUT.mkdir()
os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH']
os.environ['TORCH_CUDA_ARCH_LIST']='8.9';os.environ['MAX_JOBS']='1'
manifest=json.loads((ROOT/'INPUT_MANIFEST.json').read_text())
for p,sha in manifest.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha,p
affinity=sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0,{affinity[0]})
import numpy as np
import torch
torch.set_num_threads(1)
import triton
from torch.utils.cpp_extension import load
PROJECT=Path('03_hadamard_tc/a962695448-rgb')
sys.path.insert(0,str(ROOT/'candidate/cuda'/PROJECT/'scripts'))
from verify_paired_quantization import verify as pair_checks
from verify_singleton_quantization import verify as singleton_checks
from verify_quantize_packed import verify as packed_checks,contexts
from verify_out_buffers import buffers_for,positive_cases,negative_cases,mutation_contract
from verify_execution_context import compare
protocol=json.loads((ROOT/'PROTOCOL-CONFIRMATION.json').read_text())['cuda']
summary={'status':'RUNNING','jobs':[],'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'triton':triton.__version__,
    'cpu_affinity_before':affinity,'cpu_affinity_pinned':sorted(os.sched_getaffinity(0)),'torch_cpu_threads':torch.get_num_threads(),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
    'nvcc':subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True),'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)},
    'manifest_sha256':hashlib.sha256((ROOT/'INPUT_MANIFEST.json').read_bytes()).hexdigest(),
    'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL-CONFIRMATION.json').read_bytes()).hexdigest()}


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2)+'\n')


def module(version):
    root=ROOT/version/'cuda'/PROJECT;build=ROOT/'build'/version;build.mkdir(parents=True,exist_ok=True)
    return load(name='hadamard_pairs_'+version,sources=[str(root/'src/torch_binding.cu')],extra_include_paths=[str(root/'include')],
      extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17','-lineinfo','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
      '-U__CUDA_NO_BFLOAT16_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','--expt-relaxed-constexpr'],build_directory=str(build),with_cuda=True,verbose=True)


def configurations():
    for rows in protocol['target_rows']:
        for dim in protocol['target_dims']:yield 'packed_quant',rows,dim,True
    for rows in protocol['control_rows']:
        for dim in protocol['target_dims']:yield 'packed_quant',rows,dim,False
    yield 'packed_quant',65536,1,False
    for kind in ('original_quant','packed_hadamard','packed_fused'):yield kind,65536,8,False


def call(op,x,kind,threads,buffers=None):
    name={'packed_quant':'quantize_int4_packed','original_quant':'quantize_int4','packed_hadamard':'hadamard','packed_fused':'hadamard_int4'}[kind]
    options={'block_threads':threads}
    if kind in ('packed_hadamard','packed_fused'):options.update(scale=1.,row_layout='packed')
    return functools.partial(getattr(op,name+('_out' if buffers is not None else '')),x,*(buffers or ()),**options)


def graph_for(fn):
    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(30):fn()
    side.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=side):
        for _ in range(protocol['graph_nodes']):fn()
    torch.cuda.synchronize()
    for _ in range(10):graph.replay()
    torch.cuda.synchronize()
    return graph


def device_time(graph):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(protocol['graph_replays']):graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)*1000/(protocol['graph_nodes']*protocol['graph_replays'])


def host_time(fn):
    torch.cuda.synchronize();start=time.perf_counter_ns()
    for _ in range(protocol['host_calls']):fn()
    torch.cuda.synchronize()
    return (time.perf_counter_ns()-start)/1000/protocol['host_calls']


save('summary.json',summary)
try:
    control,candidate=module('control'),module('candidate')
    summary['binaries']={v:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for v,m in [('control',control),('candidate',candidate)]}
    summary['correctness_reused']='v2 unchanged source, all original correctness checks passed'
    report={'status':'RUNNING','records':[],'host_records':[],'groups':[],'calibration':[],'protocol':protocol}
    save('benchmark.json',report)
    for round_id in range(protocol['rounds']):
        rng=torch.Generator().manual_seed(9172030+round_id)
        for dtype in (torch.float16,torch.bfloat16):
            for kind,rows,dim,target in configurations():
                x=torch.randn((rows,dim),generator=rng).to(dtype).cuda()
                buffer_kind='hadamard' if kind=='packed_hadamard' else 'quantize_int4'
                for threads in protocol['threads']:
                    buffers={v:buffers_for(torch,x,buffer_kind)[0] for v in ('control','candidate')}
                    modules={'control':control,'candidate':candidate}
                    fns={v:call(m,x,kind,threads,buffers[v]) for v,m in modules.items()}
                    expected=call(control,x,kind,threads)();expected=expected if isinstance(expected,tuple) else (expected,)
                    expected=tuple(t.cpu() for t in expected)
                    graphs={v:graph_for(fn) for v,fn in fns.items()}
                    for v in modules:fns[v]();compare(torch,buffers[v],expected)
                    timings={v:[] for v in modules}
                    for group in range(protocol['groups']):
                        order=('control','candidate') if (group+round_id)%2==0 else ('candidate','control')
                        for v in order:timings[v].append(device_time(graphs[v]))
                    ratio=statistics.median(timings['control'])/statistics.median(timings['candidate'])
                    case={'round':round_id,'dtype':str(dtype),'kind':kind,'rows':rows,'dim':dim,'threads':threads,'target':target}
                    report['records'].append(dict(case,device_us=timings,speedup=ratio,gate_passed=ratio>=1/1.05))
                    del graphs
                    if threads==256:
                        for method in ('allocating','out'):
                            funcs={v:call(m,x,kind,threads,buffers[v] if method=='out' else None) for v,m in modules.items()}
                            for fn in funcs.values():
                                for _ in range(30):fn()
                            timings={v:[] for v in modules}
                            for group in range(protocol['groups']):
                                order=('control','candidate') if (group+round_id)%2==0 else ('candidate','control')
                                for v in order:timings[v].append(host_time(funcs[v]))
                            ratio=statistics.median(timings['control'])/statistics.median(timings['candidate'])
                            report['host_records'].append(dict(case,method=method,host_us=timings,speedup=ratio,gate_passed=ratio>=1/1.05))
                save('benchmark.json',report)
        for dtype,kind,rows,dim in [(torch.float16,'packed_quant',257,2),(torch.bfloat16,'packed_hadamard',65536,8)]:
            x=torch.randn((rows,dim),generator=rng).to(dtype).cuda()
            functions={'a':call(control,x,kind,256),'b':call(control,x,kind,256)}
            for fn in functions.values():
                for _ in range(1000):fn()
            samples={v:[] for v in functions}
            for group in range(protocol['groups']):
                order=('a','b') if (group+round_id)%2==0 else ('b','a')
                for v in order:samples[v].append(host_time(functions[v]))
            ratio=statistics.median(samples['a'])/statistics.median(samples['b'])
            report['calibration'].append({'round':round_id,'dtype':str(dtype),'kind':kind,'rows':rows,'dim':dim,'host_us':samples,'same_function_ratio':ratio,'gate_passed':1/1.05<=ratio<=1.05})
        for dtype in ('torch.float16','torch.bfloat16'):
            for threads in protocol['threads']:
                records=[x for x in report['records'] if x['round']==round_id and x['dtype']==dtype and x['threads']==threads and x['target']]
                ratio=math.exp(statistics.mean(math.log(x['speedup']) for x in records))
                report['groups'].append({'round':round_id,'dtype':dtype,'threads':threads,'target_geomean_speedup':ratio,'gate_passed':ratio>=1.10})
        save('benchmark.json',report);print('round',round_id,'complete',flush=True)
    report['status']='ACCEPT' if all(x['gate_passed'] for x in report['records']+report['host_records']+report['groups']+report['calibration']) else 'REJECT'
    save('benchmark.json',report);summary['cuda_decision']=report['status']
except Exception:
    summary['cuda_error']=traceback.format_exc();print(summary['cuda_error'],flush=True)

nine=ROOT/'candidate/nine';env=dict(os.environ,PYTHONPATH=str(nine/'src'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
for name,args in ([] if os.environ.get('AP_CUDA_ONLY')=='1' else [('nine-gpu',['scripts/verify_interpreter_gpu.py','--report',str(OUT/'nine-gpu.json')]),
                  ('nine-address-tests',['-m','pytest','-q','tests/test_interpreter_active_addresses.py','tests/test_interpreter_memory_dependencies.py','-W','error','--junitxml='+str(OUT/'nine-address-tests.xml')])]):
    start=time.monotonic()
    with (OUT/f'{name}.log').open('x') as log:
        result=subprocess.run([sys.executable,*args],cwd=nine,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    summary['jobs'].append({'name':name,'returncode':result.returncode,'seconds':time.monotonic()-start})
for p,sha in manifest.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha,p
summary['sources_unchanged']=True
summary['status']='PASS' if 'cuda_error' not in summary and all(j['returncode']==0 for j in summary['jobs']) else 'FAIL'
save('summary.json',summary)
files={str(p.relative_to(ROOT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()} for p in OUT.iterdir() if p.is_file()}
payload=json.dumps(files,ensure_ascii=True).encode();(ROOT/'CONFIRMATION_EXPORT.txt').write_bytes(payload)
receipt={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'status':summary['status'],'cuda_decision':summary.get('cuda_decision')}
(ROOT/'CONFIRMATION_RECEIPT.txt').write_text(json.dumps(receipt));print(receipt,flush=True)
