"""Isolated same-device old/new quantize validation; retain every sample."""
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
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpu-final';OUT.mkdir()
os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH']
os.environ['TORCH_CUDA_ARCH_LIST']='8.9';os.environ['MAX_JOBS']='1'
manifest=json.loads((ROOT/'INPUT_MANIFEST.json').read_text())
for path,sha in manifest.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
import numpy as np
import torch
import triton
from torch.utils.cpp_extension import load
project=Path('03_hadamard_tc/a962695448-rgb')
sys.path.insert(0,str(ROOT/'candidate/cuda'/project/'scripts'))
from verify_quantize_packed import verify, contexts
from verify_out_buffers import buffers_for, positive_cases, negative_cases, mutation_contract
from verify_execution_context import compare
protocol=json.loads((ROOT/'PROTOCOL-CUDA-FINAL.json').read_text())['cuda']
summary={'status':'RUNNING','jobs':[],'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'triton':triton.__version__,'numpy':np.__version__,
         'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'arch_list':os.environ['TORCH_CUDA_ARCH_LIST'],
         'nvcc':subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True),'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)},
         'manifest_sha256':hashlib.sha256((ROOT/'INPUT_MANIFEST.json').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL-CUDA-FINAL.json').read_bytes()).hexdigest(),
         'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2)+'\n')


def module(version):
    root=ROOT/version/'cuda'/project;build=ROOT/'build'/version;build.mkdir(parents=True,exist_ok=True)
    flags=['-O3','-std=c++17','-lineinfo','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
           '-U__CUDA_NO_BFLOAT16_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','--expt-relaxed-constexpr']
    return load(name='hadamard_quant_'+version,sources=[str(root/'src/torch_binding.cu')],extra_include_paths=[str(root/'include')],
                extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=flags,build_directory=str(build),with_cuda=True,verbose=True)


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
    summary['extension_sha256']={v:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for v,m in [('control',control),('candidate',candidate)]}
    correctness=verify(torch,np,candidate,control);save('packed-correctness.json',correctness)
    correctness['contexts']=contexts(torch,np,candidate);save('packed-correctness.json',correctness)
    print('packed correctness PASS',len(correctness['positive']),len(correctness['contexts']),flush=True)
    legacy={'positive':positive_cases(torch,np,candidate),'negative':negative_cases(torch,candidate),'mutation':mutation_contract(torch,np,candidate),'status':'PASS'}
    save('legacy-correctness.json',legacy);print('legacy correctness PASS',flush=True)
    report={'status':'RUNNING','records':[],'groups':[],'host_records':[],'protocol':protocol}
    save('quantize-benchmark.json',report)
    for round_id in range(protocol['rounds']):
        rng=torch.Generator().manual_seed(9172026+round_id)
        for dtype in (torch.float16,torch.bfloat16):
            for rows in protocol['target_rows']:
                for dim in protocol['target_dims']:
                    x=torch.randn((rows,dim),generator=rng).to(dtype).cuda()
                    expected=tuple(v.cpu() for v in control.quantize_int4(x))
                    for threads in protocol['threads']:
                        buffers={n:buffers_for(torch,x,'quantize_int4')[0] for n in ('control','original','packed')}
                        fns={
                            'control':functools.partial(control.quantize_int4_out,x,*buffers['control'],threads),
                            'original':functools.partial(candidate.quantize_int4_out,x,*buffers['original'],threads),
                            'packed':functools.partial(candidate.quantize_int4_packed_out,x,*buffers['packed'],threads),
                        }
                        graphs={name:graph_for(fn) for name,fn in fns.items()}
                        for name in fns:fns[name]();compare(torch,buffers[name],expected)
                        samples={name:[] for name in fns}
                        for group in range(protocol['groups']):
                            order=list(fns);shift=(group+round_id)%3;order=order[shift:]+order[:shift]
                            for name in order:samples[name].append(device_time(graphs[name]))
                        ratio=statistics.median(samples['control'])/statistics.median(samples['packed'])
                        original=statistics.median(samples['control'])/statistics.median(samples['original'])
                        report['records'].append({'round':round_id,'dtype':str(dtype),'rows':rows,'dim':dim,'threads':threads,'device_us':samples,
                            'speedup':ratio,'legacy_speedup':original,'gate_passed':min(ratio,original)>=1/1.05})
                        del graphs
                        if threads==256:
                            for method in ('allocating','out'):
                                calls=fns if method=='out' else {
                                  'control':functools.partial(control.quantize_int4,x,threads),
                                  'original':functools.partial(candidate.quantize_int4,x,threads),
                                  'packed':functools.partial(candidate.quantize_int4_packed,x,threads)}
                                for fn in calls.values():
                                    for _ in range(30):fn()
                                host={name:[] for name in calls}
                                for group in range(protocol['groups']):
                                    order=list(calls);shift=(group+round_id)%3;order=order[shift:]+order[:shift]
                                    for name in order:host[name].append(host_time(calls[name]))
                                report['host_records'].append({'round':round_id,'dtype':str(dtype),'rows':rows,'dim':dim,'threads':threads,'method':method,'host_us':host,
                                    'speedup':statistics.median(host['control'])/statistics.median(host['packed']),
                                    'legacy_speedup':statistics.median(host['control'])/statistics.median(host['original'])})
                save('quantize-benchmark.json',report)
        for dtype in ('torch.float16','torch.bfloat16'):
            for threads in protocol['threads']:
                records=[r for r in report['records'] if r['round']==round_id and r['dtype']==dtype and r['threads']==threads]
                ratio=math.exp(statistics.mean(math.log(r['speedup']) for r in records))
                report['groups'].append({'round':round_id,'dtype':dtype,'threads':threads,'geomean_speedup':ratio,'gate_passed':ratio>=1.10})
        save('quantize-benchmark.json',report);print('round',round_id,'finished',flush=True)
    report['status']='ACCEPT' if all(r['gate_passed'] for r in report['records']+report['groups']) and all(min(r['speedup'],r['legacy_speedup'])>=1/1.05 for r in report['host_records']) else 'REJECT'
    save('quantize-benchmark.json',report);summary['cuda_decision']=report['status']
except Exception:
    summary['cuda_error']=traceback.format_exc();print(summary['cuda_error'],flush=True)

nine=ROOT/'candidate/nine';env=dict(os.environ,PYTHONPATH=str(nine/'src'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
for name,args in ([] if os.environ.get('MQ_CUDA_ONLY')=='1' else [('nine-gpu',['scripts/verify_interpreter_gpu.py','--report',str(OUT/'nine-gpu.json')]),
                  ('nine-identity',['-m','pytest','-q','tests/test_interpreter_identity_memory.py','-W','error','--junitxml='+str(OUT/'nine-identity.xml')])]):
    start=time.monotonic()
    with (OUT/f'{name}.log').open('x') as log:
        job=subprocess.run([sys.executable,*args],cwd=nine,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    summary['jobs'].append({'name':name,'returncode':job.returncode,'seconds':time.monotonic()-start})
for path,sha in manifest.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
summary['sources_unchanged']=True
summary['status']='PASS' if 'cuda_error' not in summary and all(j['returncode']==0 for j in summary['jobs']) else 'FAIL'
save('summary.json',summary)
files={f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'text':f.read_text()} for f in OUT.iterdir() if f.is_file()}
payload=json.dumps(files,ensure_ascii=True);(ROOT/'GPU_FINAL_EXPORT.txt').write_text(payload)
receipt={'bytes':len(payload.encode()),'sha256':hashlib.sha256(payload.encode()).hexdigest(),'status':summary['status'],'cuda_decision':summary.get('cuda_decision')}
(ROOT/'FINAL_EXPORT_RECEIPT.txt').write_text(json.dumps(receipt));print(receipt,flush=True)
