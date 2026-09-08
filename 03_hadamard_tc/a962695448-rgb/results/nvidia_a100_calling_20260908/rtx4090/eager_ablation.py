from pathlib import Path
import argparse,gc,hashlib,json,math,random,statistics,sys,time,traceback
ROOT=Path(__file__).resolve().parent;PROD=ROOT/'production';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PROD/'project/scripts'))
import measurement_helpers as measure
import compare_reference as reference
from build_torch_extension import load_extension
p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True,choices=(1,2,3));a=p.parse_args()
protocol=json.loads((ROOT/'eager_protocol.json').read_text());out=ROOT/'eager'/f'round{a.round}.json';out.parent.mkdir(exist_ok=True);assert not out.exists()
r={'status':'RUNNING','round':a.round,'started_utc':measure.utc(),'protocol_sha256':measure.sha(ROOT/'eager_protocol.json'),'script_sha256':measure.sha(Path(__file__)),'cases':[]}
def save():out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
try:
 v=PROD/'runs/validation/report.json';assert json.loads(v.read_text())['status']=='PASS';r['validation_sha256']=measure.sha(v)
 manifest=json.loads((PROD/'production_manifest.json').read_text());assert manifest['source_commit']==protocol['source_commit']
 for name,item in manifest['files'].items():assert measure.sha(PROD/name)==item['sha256']
 import torch
 assert torch.cuda.get_device_name()==protocol['gpu'];r['gpu']=torch.cuda.get_device_name();r['before']=measure.snapshot()
 op=load_extension(verbose=True,build_directory=str(PROD/'build/production'));r['binary_sha256']=measure.sha(op.__file__)
 cases=list(protocol['cases']);random.Random(protocol['seed']+a.round).shuffle(cases);r['case_order']=cases;timing=protocol['timing']
 with torch.inference_mode():
  for ci,c in enumerate(cases):
   dtype=torch.float16 if c['dtype']=='fp16' else torch.bfloat16;x=reference.make_input(torch,[c['rows'],c['dim']],dtype,'normal',protocol['seed'],'cuda');before=x.clone();fn=getattr(op,c['mode']);expected=fn(x,1.,128)
   if c['mode']=='hadamard':
    funcs={'original_pos3':lambda:fn(x,1.,128),'original_pos_full':lambda:fn(x,1.,128,'original'),'auto_pos_full':lambda:fn(x,1.,128,'auto'),'packed_pos_full':lambda:fn(x,1.,128,'packed'),'original_kwrow':lambda:fn(x,1.,128,row_layout='original'),'auto_kwrow':lambda:fn(x,1.,128,row_layout='auto'),'packed_kwrow':lambda:fn(x,1.,128,row_layout='packed')}
   else:
    funcs={'original_pos3':lambda:fn(x,1.,128),'original_pos_full':lambda:fn(x,1.,128,'original','original'),'auto_pos_full':lambda:fn(x,1.,128,'original','auto'),'packed_pos_full':lambda:fn(x,1.,128,'original','packed'),'original_kwrow':lambda:fn(x,1.,128,row_layout='original'),'auto_kwrow':lambda:fn(x,1.,128,row_layout='auto'),'packed_kwrow':lambda:fn(x,1.,128,row_layout='packed')}
   assert set(funcs)==set(protocol['methods'])
   for name,f in funcs.items():measure.exact(torch,f(),expected,'before '+name)
   samples={name:[] for name in funcs};intervals={name:[] for name in funcs};orders=measure.orders_for(funcs,a.round,ci,timing['groups'])
   for order in orders:
    for name in order:
     f=funcs[name]
     for _ in range(timing['warmup_calls_per_group']):f()
     torch.cuda.synchronize();start=time.perf_counter_ns()
     for _ in range(timing['calls_per_group']):last=f()
     torch.cuda.synchronize();ns=time.perf_counter_ns()-start
     assert ns>0;intervals[name].append(ns);samples[name].append(ns/timing['calls_per_group']/1e6)
     measure.exact(torch,last,expected,'after '+name)
   measure.exact(torch,x,before,'input unchanged')
   r['cases'].append({**c,'samples_ms':samples,'raw_batch_ns':intervals,'median_ms':{k:statistics.median(v) for k,v in samples.items()},'group_order':orders,'outputs_bitwise_equal':True,'input_unchanged':True});save();gc.collect()
 r['status']='PASS'
except Exception as e:r.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());print(r['traceback'],flush=True)
r.update(finished_utc=measure.utc(),after=measure.snapshot());save();print(json.dumps({'status':r['status'],'cases':len(r['cases']),'round':a.round}),flush=True)
