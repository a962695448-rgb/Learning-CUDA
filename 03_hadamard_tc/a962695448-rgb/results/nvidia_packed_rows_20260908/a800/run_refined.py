"""Frozen small-N packing experiment: exactness first, then independent Graph rounds."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'project/scripts'))
import compare_reference as reference
from build_torch_extension import load_extension
import measurement_helpers as measure

def verify_sources():
    m=json.loads((ROOT/'experiment_manifest.json').read_text())
    for name,item in m['files'].items():
        data=(ROOT/name).read_bytes()
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],name
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('refined',),required=True);p.add_argument('--round',type=int,choices=(1,2,3),default=1);p.add_argument('--out',type=Path,required=True);p.add_argument('--reference-repo',type=Path,required=True);a=p.parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','stage':a.stage,'round':a.round,'started_utc':measure.utc(),'checks':[],'benchmarks':[],'commands':[]}
    def save():(a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    code=1;x=None
    try:
        manifest=verify_sources();report['manifest_sha256']=measure.sha(ROOT/'experiment_manifest.json');report['base_commit']=manifest['base_commit'];report['before']=measure.snapshot();save()
        import numpy as np
        import torch
        from torch.utils.cpp_extension import load
        import fast_hadamard_transform as dao
        import fast_hadamard_transform_cuda as dao_backend
        report['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'sm':torch.cuda.get_device_capability(),'python':sys.version}
        assert torch.cuda.device_count()==1 and list(torch.cuda.get_device_capability()) in ([8,0],[8,9])
        report['reference']=reference.provenance(dao,dao_backend,a.reference_repo)
        base=load_extension(verbose=True,build_directory=str(ROOT/'build/base'))
        build=ROOT/'build/candidate';build.mkdir(parents=True,exist_ok=True)
        candidate=load(name='infinitensor_packed_rows_experiment',sources=[str(ROOT/'candidate_binding.cu')],extra_include_paths=[str(ROOT),str(ROOT/'project/include')],extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17','-lineinfo','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_BFLOAT16_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','--expt-relaxed-constexpr'],build_directory=str(build),with_cuda=True,verbose=True)
        report['binaries']={name:{'file':module.__file__,'sha256':measure.sha(module.__file__)} for name,module in [('base',base),('candidate',candidate)]}
        protocol=json.loads((ROOT/'experiment_protocol.json').read_text())
        if a.stage!='correctness':
            gate=json.loads((ROOT/'runs/correctness/report.json').read_text());assert gate['status']=='PASS'
            assert gate['manifest_sha256']==report['manifest_sha256'] and gate['binaries']==report['binaries']
        refinement=json.loads((ROOT/'refinement_protocol.json').read_text())
        assert measure.sha(Path(__file__))==refinement['script_sha256']
        report['refinement_protocol_sha256']=measure.sha(ROOT/'refinement_protocol.json')
        cases=[{'dtype':d,'rows':m,'dim':r['dim'],'scale':scale,'mode':r['mode']} for r in refinement['rules'] for d in ('fp16','bf16') for m in refinement['rows'] if m>=r['min_rows'] or m==(4093 if r['mode']=='hadamard' else 253) for scale in ([1.] if r['dim']==1 else [1.,1/math.sqrt(r['dim'])])]
        random.Random(20260908+a.round).shuffle(cases);report['case_order']=cases
        with torch.inference_mode():
            if a.stage=='correctness':
                for case in cases:
                    dtype=torch.float16 if case['dtype']=='fp16' else torch.bfloat16;n=case['dim'];m=case['rows'];scale=case['scale']
                    entry={**case,'conditions':0};report['checks'].append(entry)
                    for pattern,seed in [('normal',2026),('normal',95811),('outlier',2026),('zeros',2026)]:
                        values=reference.make_input(torch,[m,n],dtype,pattern,seed,'cuda')
                        for offset in (0,2):
                            prefix=8 if offset==0 else 1;storage=torch.full((m*n+16,),123,dtype=dtype,device='cuda');x=storage[prefix:prefix+m*n].view(m,n);x.copy_(values);before=storage.clone()
                            assert x.data_ptr()%16==offset
                            report['active']={**case,'pattern':pattern,'seed':seed,'offset':offset}
                            expected=base.hadamard(x,scale,128);quantized=base.hadamard_int4(x,scale,128)
                            ref=dao.hadamard_transform(x if offset==0 else x.clone(),scale)
                            metrics=reference.metrics(torch,expected,ref,.01 if dtype==torch.float16 else .05);assert metrics['pass']
                            for threads in (128,256):
                                measure.exact(torch,candidate.hadamard(x,scale,threads),expected,'transform vs base')
                                measure.exact(torch,candidate.hadamard_int4(x,scale,threads),quantized,'fused vs base')
                                measure.exact(torch,candidate.quantize_int4(expected,threads),quantized,'quantize vs split')
                            y=expected.float().cpu().numpy();scales=np.max(np.abs(y),axis=1).astype(np.float32)/np.float32(7);scales[scales==0]=1
                            q=np.clip(np.rint(y/scales[:,None]),-7,7).astype(np.int8);packed=q[:,0::2].astype(np.uint8)&15
                            if n>1:packed|=(q[:,1::2].astype(np.uint8)&15)<<4
                            assert quantized[0].cpu().numpy().tobytes()==packed.tobytes() and quantized[1].cpu().numpy().tobytes()==scales.tobytes()
                            measure.exact(torch,storage,before,'input and guards');entry['conditions']+=1
                    save();print('CHECKED',json.dumps(case),flush=True)
                # A separate four-dimensional, offset input on a non-default stream.
                for n in (1,2,4,8,16):
                    for dtype in (torch.float16,torch.bfloat16):
                        stream=torch.cuda.Stream()
                        with torch.cuda.stream(stream):
                            storage=torch.arange(3*7*n+16,device='cuda',dtype=torch.float32).to(dtype);x=storage[1:1+3*7*n].view(1,3,7,n)
                            before=storage.clone();y=base.hadamard(x,.25);q=base.hadamard_int4(x,.25)
                            for threads in (128,256):
                                measure.exact(torch,candidate.hadamard(x,.25,threads),y,'4D stream transform')
                                measure.exact(torch,candidate.hadamard_int4(x,.25,threads),q,'4D stream fused')
                            measure.exact(torch,storage,before,'4D guards')
                        stream.synchronize()
                report['four_dimensional_stream_conditions']=10
            else:
                for i,case in enumerate(cases):
                    dtype=torch.float16 if case['dtype']=='fp16' else torch.bfloat16
                    x=reference.make_input(torch,[case['rows'],case['dim']],dtype,'normal',9090807,'cuda');scale=case['scale']
                    for mode in (case['mode'],):
                        funcs={'original128':lambda:getattr(base,mode)(x,scale,128),'original256':lambda:getattr(base,mode)(x,scale,256),'packed128':lambda:getattr(candidate,mode)(x,scale,128),'packed256':lambda:getattr(candidate,mode)(x,scale,256)}
                        expected=funcs['original128']()
                        for f in funcs.values():measure.exact(torch,f(),expected,'timed methods')
                        timing=measure.measure_graph(torch,funcs,a.round,i,protocol['timing'])
                        report['benchmarks'].append({**case,'mode':mode,**timing});gc.collect();save()
                    print('TIMED',json.dumps(case),flush=True)
        verify_sources();report.update(status='PASS',configuration_count=len(cases));code=0
    except Exception as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());print(report['traceback'],flush=True)
        if x is not None:
            try:
                import numpy as np
                np.savez_compressed(a.out/'failure_input.npz',bits=x.cpu().contiguous().view(torch.int16).numpy().view(np.uint16),shape=np.array(x.shape))
            except Exception as capture_error:report['input_capture_error']=repr(capture_error)
    report.update(exit_code=code,finished_utc=measure.utc(),after=measure.snapshot());save();return code

if __name__=='__main__':raise SystemExit(main())
