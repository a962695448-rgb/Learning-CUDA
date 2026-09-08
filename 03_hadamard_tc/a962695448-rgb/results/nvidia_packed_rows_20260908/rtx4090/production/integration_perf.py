import argparse,gc,hashlib,json,math,os,random,statistics,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'project/scripts'));sys.path.insert(0,str(ROOT.parent))
from build_torch_extension import load_extension
import compare_reference as reference
import measurement_helpers as measure
p=argparse.ArgumentParser();p.add_argument('--round',type=int,choices=(1,2,3),required=True);a=p.parse_args();out=ROOT/f'runs/performance{a.round}.json';assert not out.exists()
report={'status':'RUNNING','round':a.round,'started_utc':measure.utc(),'benchmarks':[],'eager':[]}
def save():out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
try:
    validation=json.loads((ROOT/'runs/validation/report.json').read_text());assert validation['status']=='PASS'
    report['validation_sha256']=measure.sha(ROOT/'runs/validation/report.json')
    m=json.loads((ROOT/'production_manifest.json').read_text())
    for n,item in m['files'].items():assert measure.sha(ROOT/n)==item['sha256']
    import torch
    op=load_extension(verbose=True,build_directory=str(ROOT/'build/production'))
    report['binary_sha256']=measure.sha(op.__file__);report['gpu']=torch.cuda.get_device_name();report['before']=measure.snapshot()
    policy=json.loads((ROOT.parent/'runs/final_policy.json').read_text());assert policy['gpu']==report['gpu']
    report['policy_sha256']=measure.sha(ROOT.parent/'runs/final_policy.json')
    cases=[]
    for r in policy['rules']:
        for rows in sorted({r['min_rows'],r['min_rows']+1,max(4096,r['min_rows']),16384}):
            for dtype in ('fp16','bf16'):
                cases.append({'dim':r['dim'],'mode':r['mode'],'rows':rows,'dtype':dtype,'scale':1.,'activated':True})
    for n,rows,mode in [(16,65,'hadamard_int4'),(16,4096,'hadamard_int4'),(32,17,'hadamard'),(64,65,'hadamard_int4'),(256,17,'hadamard')]:
        for dtype in ('fp16','bf16'):cases.append({'dim':n,'mode':mode,'rows':rows,'dtype':dtype,'scale':1.,'activated':False})
    random.Random(991000+a.round).shuffle(cases);report['case_order']=cases
    timing=json.loads((ROOT.parent/'experiment_protocol.json').read_text())['timing']
    with torch.inference_mode():
        for index,c in enumerate(cases):
            dtype=torch.float16 if c['dtype']=='fp16' else torch.bfloat16;x=reference.make_input(torch,[c['rows'],c['dim']],dtype,'normal',2026,'cuda');mode=c['mode'];scale=c['scale']
            funcs={'original128':lambda:getattr(op,mode)(x,scale,128),'original256':lambda:getattr(op,mode)(x,scale,256),'auto':lambda:getattr(op,mode)(x,scale,128,row_layout='auto')}
            expected=funcs['original128']()
            for f in funcs.values():measure.exact(torch,f(),expected,'integrated auto')
            result=measure.measure_graph(torch,funcs,a.round,index,timing);report['benchmarks'].append({**c,**result});save();gc.collect()
        for n,rows,mode in [(1,1,'hadamard_int4'),(4,17,'hadamard_int4'),(16,4096,'hadamard'),(16,65,'hadamard_int4')]:
            for dtype_name in ('fp16','bf16'):
                dtype=torch.float16 if dtype_name=='fp16' else torch.bfloat16;x=reference.make_input(torch,[rows,n],dtype,'normal',2026,'cuda')
                funcs={'original128':lambda:getattr(op,mode)(x,1.,128),'auto':lambda:getattr(op,mode)(x,1.,128,row_layout='auto')};values={k:[] for k in funcs}
                for g in range(5):
                    order=list(funcs) if (g+a.round)%2 else list(reversed(funcs))
                    for name in order:
                        for _ in range(25):funcs[name]()
                        torch.cuda.synchronize();start=time.perf_counter_ns()
                        for _ in range(500):last=funcs[name]()
                        torch.cuda.synchronize();values[name].append((time.perf_counter_ns()-start)/500/1e6)
                report['eager'].append({'dim':n,'rows':rows,'mode':mode,'dtype':dtype_name,'samples_ms':values,'median_ms':{k:statistics.median(v) for k,v in values.items()},'scope':'Host wall time per eager allocating API call; synchronization brackets the 500-call batch. Separate from Graph timings.'})
    report['status']='PASS'
except Exception as e:report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());print(report['traceback'],flush=True)
report.update(finished_utc=measure.utc(),after=measure.snapshot());save();print(json.dumps({'status':report['status'],'benchmarks':len(report['benchmarks']),'eager':len(report['eager'])}),flush=True)
