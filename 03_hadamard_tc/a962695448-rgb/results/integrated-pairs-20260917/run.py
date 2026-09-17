"""Validate both launch frontends before three fresh full-matrix timing runs."""
import base64
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results-remote';OUT.mkdir(exist_ok=False)
PROJECT=Path('03_hadamard_tc/a962695448-rgb')
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MAX_JOBS='1',TORCH_CUDA_ARCH_LIST='8.9')
os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH']
manifest=json.loads((ROOT/'INPUT_MANIFEST.json').read_text())
for path,sha in manifest.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[0]})
import numpy as np
import torch
torch.set_num_threads(1)
from torch.utils.cpp_extension import load
sys.path.insert(0,str(ROOT/'candidate/cuda'/PROJECT/'scripts'))
from verify_hadamard_packed import verify as pair_checks,contexts as paired_contexts
from verify_paired_quantization import verify as quantization_pair_checks
from verify_singleton_quantization import verify as singleton_checks
from verify_quantize_packed import verify as packed_checks,contexts as quantization_contexts
from verify_out_buffers import positive_cases,negative_cases,mutation_contract,execution_contexts

summary={'status':'RUNNING','jobs':[],'environment':{'python':sys.version,'numpy':np.__version__,
  'torch':torch.__version__,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
  'capability':list(torch.cuda.get_device_capability()),'cpu_affinity_before':affinity,
  'cpu_affinity_pinned':sorted(os.sched_getaffinity(0)),'torch_cpu_threads':torch.get_num_threads(),
  'nvcc':subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True),
  'nvidia_smi_before':subprocess.check_output(['nvidia-smi'],text=True)},
  'manifest_sha256':hashlib.sha256((ROOT/'INPUT_MANIFEST.json').read_bytes()).hexdigest(),
  'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest()}


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2)+'\n')


def job(name,command,cwd,timeout=600):
    start=time.monotonic()
    with (OUT/(name+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=cwd,env=os.environ,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    summary['jobs'].append({'name':name,'command':command,'returncode':result.returncode,'seconds':time.monotonic()-start})
    save('summary.json',summary)
    assert result.returncode==0,(name,result.returncode)
    print(name,'PASS',flush=True)
    return (OUT/(name+'.log')).read_text()


def module(label,source,name):
    project=ROOT/source/'cuda'/PROJECT;build=ROOT/'build'/label;build.mkdir(parents=True)
    return load(name=name,sources=[str(project/'src/torch_binding.cu')],extra_include_paths=[str(project/'include')],
      extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17','-lineinfo',
      '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_BFLOAT16_OPERATORS__',
      '-U__CUDA_NO_BFLOAT16_CONVERSIONS__','--expt-relaxed-constexpr'],build_directory=str(build),with_cuda=True,verbose=True)


save('summary.json',summary)
try:
    for version in ('control','candidate'):
        project=ROOT/version/'cuda'/PROJECT
        job('cli-build-'+version,['make','all','NVCC=/usr/local/cuda/bin/nvcc','ARCH=89'],project)
        label='integrated_'+version
        job('cli-validation-'+version,[sys.executable,'scripts/run_validation.py','--label',label],project)
        validation=project/'results'/f'validation_{label}.log'
        text=validation.read_text();assert 'SELF_TEST PASS cases=1876' in text
        assert len(re.findall(r'EXIT_CODE 2;',text))==15
        shutil.copy2(validation,OUT/f'cli-validation-{version}-details.log')
        modes=[('packed',256)] if version=='control' else [('original',256),('packed',128),('packed',256),('auto',128),('auto',256)]
        for layout,threads in modes:
            text=job(f'cli-selftest-{version}-{layout}-{threads}',[str(project/'build/hadamard'),'--self-test','--row-layout',layout,'--block-threads',str(threads)],project)
            assert 'SELF_TEST PASS cases=1876' in text
    job('cpu-reference-and-row-policy',['make','cpu-test'],ROOT/'candidate/cuda'/PROJECT)
    control=module('control_a','control','ip_control_a')
    twin=module('control_b','control','ip_control_b')
    candidate=module('candidate','candidate','ip_candidate')
    summary['binaries']={label:hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest() for label,op in [('control',control),('twin',twin),('candidate',candidate)]}
    save('pairs-candidate.json',pair_checks(torch,np,candidate,control))
    save('pairs-control.json',pair_checks(torch,np,control,control))
    save('pairs-twin.json',pair_checks(torch,np,twin,control))
    save('paired-contexts.json',{'status':'PASS','cases':paired_contexts(torch,np,candidate)})
    save('quantization-pairs.json',quantization_pair_checks(torch,np,candidate,control))
    save('singleton.json',singleton_checks(torch,np,candidate,control))
    job('overflow-compatibility',[sys.executable,'-u',str(ROOT/'tools/verify_overflow_pairs.py'),str(ROOT)],ROOT,180)
    packed=packed_checks(torch,np,candidate,control);packed['contexts']=quantization_contexts(torch,np,candidate);save('packed-regression.json',packed)
    save('legacy-regression.json',{'status':'PASS','positive':positive_cases(torch,np,candidate),'negative':negative_cases(torch,candidate),
      'mutation':mutation_contract(torch,np,candidate),'contexts':execution_contexts(torch,np,candidate)})
    print('ALL_CORRECTNESS_PASS',flush=True)
    for round_id in range(3):
        job(f'timing-round-{round_id}',[sys.executable,'-u',str(ROOT/'bench.py'),'--round',str(round_id)],ROOT,1800)
    reports=[json.loads((OUT/f'round-{i}.json').read_text()) for i in range(3)]
    combined={'status':'ACCEPT' if all(r['status']=='ACCEPT' for r in reports) else 'REJECT',
      'protocol':json.loads((ROOT/'PROTOCOL.json').read_text())['cuda']}
    for key in ('records','host_records','groups','calibration'):combined[key]=[item for report in reports for item in report[key]]
    save('benchmark.json',combined);summary['candidate_decision']=combined['status']
    summary['status']='PASS'
except Exception:
    summary['status']='FAIL';summary['error']=traceback.format_exc();print(summary['error'],flush=True)
for path,sha in manifest.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
summary['sources_unchanged']=True
summary['environment']['nvidia_smi_after']=subprocess.check_output(['nvidia-smi'],text=True)
save('summary.json',summary)
files={str(p.relative_to(ROOT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()} for p in OUT.iterdir() if p.is_file()}
payload=json.dumps(files,ensure_ascii=True).encode()
compressed=gzip.compress(payload,mtime=0);encoded=base64.b64encode(compressed).decode('ascii')
(ROOT/'FINAL_EXPORT.base64.txt').write_text(encoded)
parts=[]
for i,start in enumerate(range(0,len(encoded),400000)):
    name=f'FINAL_EXPORT_PART_{i:02d}.txt';part=encoded[start:start+400000];(ROOT/name).write_text(part)
    parts.append({'name':name,'characters':len(part),'sha256':hashlib.sha256(part.encode()).hexdigest()})
receipt={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'gzip_bytes':len(compressed),
  'gzip_sha256':hashlib.sha256(compressed).hexdigest(),'base64_characters':len(encoded),'parts':parts,
  'status':summary['status'],'decision':summary.get('candidate_decision')}
(ROOT/'FINAL_EXPORT_RECEIPT.txt').write_text(json.dumps(receipt));print(receipt,flush=True)
