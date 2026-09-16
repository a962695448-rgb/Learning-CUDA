"""Build the bounded 4090 D routing candidate and alternate paired measurements."""
import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--control',type=Path,required=True)
parser.add_argument('--candidate',type=Path,required=True)
parser.add_argument('--control-build',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--protocol',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH'];os.environ['MAX_JOBS']='1'
sys.path.insert(0,str(args.control/'scripts'))
from build_torch_extension import load_extension
from verify_execution_context import reference,compare,check_case
import numpy as np
import torch
from torch.utils.cpp_extension import load

protocol=json.loads(args.protocol.read_text())
assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090 D'
control=load_extension(build_directory=str(args.control_build))
build=args.output/'build';build.mkdir()
candidate=load(name='hadamard_4090d_candidate',sources=[str(args.candidate/'src/torch_binding.cu')],
 extra_include_paths=[str(args.candidate/'include')],extra_cflags=['-O3','-std=c++17'],
 extra_cuda_cflags=['-O3','-std=c++17','-lineinfo','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
                   '-U__CUDA_NO_BFLOAT16_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','--expt-relaxed-constexpr'],
 build_directory=str(build),with_cuda=True,verbose=True)
report={'status':'RUNNING','protocol':protocol,'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'environment':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__},
        'source_sha256':{},'correctness':[],'records':[],'contexts':[]}
for name,root,module in [('control',args.control,control),('candidate',args.candidate,candidate)]:
 report['source_sha256'][name]={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
  for f in sorted(root.rglob('*')) if f.is_file() and f.suffix in ('.cu','.cuh','.hpp') and not {'build','results'}.intersection(f.relative_to(root).parts)}
 report['source_sha256'][name]['extension']=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def save():
 (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


def invoke(module,x,method):
 return getattr(module,method)(x,scale=1.,row_layout='auto')


def expected(cpu,method):
 values=cpu.float().numpy().copy();n=values.shape[-1];stride=1
 while stride<n:
  blocks=values.reshape(-1,n//(2*stride),2*stride);left,right=blocks[...,:stride].copy(),blocks[...,stride:].copy()
  blocks[...,:stride]=left+right;blocks[...,stride:]=left-right;stride*=2
 y=torch.from_numpy(values).to(cpu.dtype)
 return (y,) if method=='hadamard' else reference(torch,np,y,{'shape':y.shape,'method':'quantize_int4'})


def graph_for(module,x,method):
 side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(side):
  for _ in range(10):out=invoke(module,x,method)
 side.synchronize();graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph,stream=side):
  for _ in range(protocol['graph_ops']):out=invoke(module,x,method)
 for _ in range(5):graph.replay()
 torch.cuda.synchronize()
 return graph,out


def measure(graph):
 start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 start.record()
 for _ in range(protocol['graph_replays']):graph.replay()
 end.record();end.synchronize()
 return start.elapsed_time(end)*1000/(protocol['graph_ops']*protocol['graph_replays'])


save()
for round_id in range(protocol['paired_rounds']):
 rng=torch.Generator().manual_seed(protocol['seed']+round_id)
 for dtype in (torch.float16,torch.bfloat16):
  for rows in sorted(protocol['target_rows']+protocol['control_rows']):
   for dim in protocol['dimensions']:
    cpu=torch.randn((rows,dim),generator=rng).to(dtype);x=cpu.cuda()
    for method in protocol['methods']:
     oracle=expected(cpu,method)
     for module in (control,candidate):compare(torch,invoke(module,x,method),oracle)
     compare(torch,x,(cpu,))
     key={'round':round_id,'dtype':str(dtype),'rows':rows,'dim':dim,'method':method}
     report['correctness'].append(dict(key,status='PASS'))
     graphs={'control':graph_for(control,x,method),'candidate':graph_for(candidate,x,method)}
     samples={'control':[],'candidate':[]}
     for group in range(protocol['groups']):
      order=('control','candidate') if (group+round_id)%2==0 else ('candidate','control')
      for version in order:samples[version].append(measure(graphs[version][0]))
     ratio=statistics.median(samples['control'])/statistics.median(samples['candidate'])
     target=rows in protocol['target_rows'];threshold=1.05 if target else 1/1.03
     report['records'].append(dict(key,speedup=ratio,graph_us=samples,target=target,gate_passed=ratio>=threshold))
     del graphs;gc.collect()
   save()
  print('round',round_id,str(dtype),'complete',len(report['records']),flush=True)

# Check that the newly enabled route remains correct outside the default stream.
for dtype in (torch.float16,torch.bfloat16):
 for shape in ((4097,1),(4097,8),(8193,16),(1,1,4097,8)):
  for method in protocol['methods']:
   case={'shape':shape,'method':method,'row_layout':'auto','block_threads':128}
   report['contexts'].append(check_case(torch,np,candidate,case,dtype));save()
for name,root in [('control',args.control),('candidate',args.candidate)]:
 for file,sha in report['source_sha256'][name].items():
  if file!='extension':assert hashlib.sha256((root/file).read_bytes()).hexdigest()==sha,file
report['status']='ACCEPT' if all(r['gate_passed'] for r in report['records']) else 'REJECT';save()
export={f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'text':f.read_bytes().decode()}
        for f in args.output.iterdir() if f.is_file()}
payload=json.dumps(export,ensure_ascii=True);(args.output/'CONFIRM_EXPORT.txt').write_text(payload)
print(json.dumps({'status':report['status'],'export_bytes':len(payload.encode()),'export_sha256':hashlib.sha256(payload.encode()).hexdigest()}),flush=True)
