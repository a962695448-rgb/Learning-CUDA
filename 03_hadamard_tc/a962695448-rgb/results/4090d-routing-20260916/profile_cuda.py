"""Measure resident-data kernel, host API, and layout costs on one GPU."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--build',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.project/'scripts'))
os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH']
from build_torch_extension import load_extension
import torch
from verify_execution_context import reference, compare
import numpy as np

GRAPH_OPS=32
GROUPS=7
REPLAYS=20
ROWS=(1,17,256,4096,65536)
DIMS=(1,8,16,64,256)


def invoke(op,x,method,layout,threads):
    if method=='hadamard':return op.hadamard(x,1.,threads,layout)
    if method=='quantize_int4':return op.quantize_int4(x,threads)
    return op.hadamard_int4(x,1.,threads,'contiguous256' if layout=='contiguous256' else 'original',
                            'original' if layout=='contiguous256' else layout)


def benchmark(fn):
    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(10):out=fn()
    side.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=side):
        for _ in range(GRAPH_OPS):out=fn()
    for _ in range(5):graph.replay()
    torch.cuda.synchronize()
    groups=[]
    for _ in range(GROUPS):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPLAYS):graph.replay()
        end.record();end.synchronize()
        groups.append(start.elapsed_time(end)*1000/(REPLAYS*GRAPH_OPS))
    del graph,out;gc.collect()
    host=[]
    for _ in range(3):
        torch.cuda.synchronize();start=time.perf_counter_ns()
        for _ in range(200):out=fn()
        torch.cuda.synchronize();host.append((time.perf_counter_ns()-start)/200/1000)
    return groups,host


args.output.mkdir(parents=True,exist_ok=False)
op=load_extension(verbose=True,build_directory=str(args.build))
report={'status':'RUNNING','cases':[],'method':'32-op CUDA Graph / 20 replays / 7 groups; ordinary allocating Python API / 200 calls / 3 groups',
        'scope':'Warm resident inputs and outputs; no DRAM-bandwidth or model end-to-end claim.',
        'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,
                       'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
                       'numpy':np.__version__,'extension_sha256':hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest()},
        'source_sha256':{str(f.relative_to(args.project)):hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(args.project.rglob('*')) if f.is_file() and f.suffix in ('.cu','.cuh','.hpp') and not {'build','results'}.intersection(f.relative_to(args.project).parts)}}


def save():
    (args.output/'profile.json').write_text(json.dumps(report,indent=2)+'\n')


save()
rng=torch.Generator().manual_seed(20260916)
for dtype in (torch.float16,torch.bfloat16):
    for rows in ROWS:
        for dim in DIMS:
            cpu=torch.randn((rows,dim),generator=rng).to(dtype)
            x=cpu.cuda()
            for method in ('hadamard','hadamard_int4','quantize_int4'):
                variants=[('original',128)]
                if method!='quantize_int4':variants += [('original',256),('auto',128)]
                if method!='quantize_int4' and dim<=16:variants += [('packed',128),('packed',256)]
                if method=='hadamard_int4' and dim==256:variants += [('contiguous256',128)]
                # Independent NumPy oracle uses scale 1 via the equivalent input scaling.
                oracle_case={'shape':cpu.shape,'method':method}
                if method=='quantize_int4':expected=reference(torch,np,cpu,oracle_case)
                else:
                    values=cpu.float().numpy().copy();stride=1
                    while stride<dim:
                        blocks=values.reshape(-1,dim//(2*stride),2*stride)
                        left,right=blocks[...,:stride].copy(),blocks[...,stride:].copy()
                        blocks[...,:stride]=left+right;blocks[...,stride:]=left-right;stride*=2
                    y=torch.from_numpy(values).to(dtype)
                    expected=(y,) if method=='hadamard' else reference(torch,np,y,{'shape':y.shape,'method':'quantize_int4'})
                for layout,threads in variants:
                    fn=lambda:invoke(op,x,method,layout,threads)
                    compare(torch,fn(),expected)
                    device,host=benchmark(fn)
                    report['cases'].append({'dtype':str(dtype),'rows':rows,'dim':dim,'method':method,'layout':layout,'threads':threads,
                            'graph_us':device,'graph_median_us':statistics.median(device),'host_us':host,
                            'host_median_us':statistics.median(host),'oracle_bitwise_equal':True})
            save();print(str(dtype),rows,dim,'completed',len(report['cases']),flush=True)

# A small execution trace exposes kernel names and CPU launch/allocation calls.
try:
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
    x=torch.randn((65536,256),device='cuda',dtype=torch.float16)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities,record_shapes=True) as prof:
        for _ in range(3):op.hadamard_int4(x)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.output/'torch-trace.json'))
    (args.output/'torch-profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=25))
    report['profiler']='captured'
except Exception as error:
    report['profiler_error']=f'{type(error).__name__}: {error}'
report['status']='PASS';save()
files={f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'text':f.read_bytes().decode()} for f in args.output.iterdir() if f.is_file()}
payload=json.dumps(files,ensure_ascii=True);(args.output/'PROFILE_EXPORT.txt').write_text(payload)
print(json.dumps({'status':report['status'],'export_bytes':len(payload.encode()),'export_sha256':hashlib.sha256(payload.encode()).hexdigest()}),flush=True)
