from pathlib import Path
import argparse,gc,json,random,sys,traceback
ROOT=Path(__file__).resolve().parent;PROD=ROOT/'production';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PROD/'project/scripts'))
import measurement_helpers as measure
import compare_reference as reference
from build_torch_extension import load_extension
p=argparse.ArgumentParser();p.add_argument('--round',type=int,choices=(1,2,3),required=True);a=p.parse_args();out=ROOT/f'runs/graph{a.round}.json';assert not out.exists();report={'status':'RUNNING','round':a.round,'started_utc':measure.utc(),'benchmarks':[]}
def save():out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
try:
 validation=PROD/'runs/validation/report.json';assert json.loads(validation.read_text())['status']=='PASS'
 manifest=json.loads((PROD/'production_manifest.json').read_text())
 for name,item in manifest['files'].items():assert measure.sha(PROD/name)==item['sha256']
 import torch
 policy=json.loads((ROOT/'final_policy.json').read_text());assert torch.cuda.get_device_name()==policy['gpu']
 op=load_extension(verbose=True,build_directory=str(PROD/'build/production'));report.update(gpu=torch.cuda.get_device_name(),source_commit=manifest['source_commit'],binary_sha256=measure.sha(op.__file__),validation_sha256=measure.sha(validation),policy_sha256=measure.sha(ROOT/'final_policy.json'),script_sha256=measure.sha(Path(__file__)),before=measure.snapshot())
 cases={}
 def add(dim,rows,mode,dtype,active):cases[(dim,rows,mode,dtype)]={'dim':dim,'rows':rows,'mode':mode,'dtype':dtype,'scale':1.,'activated':active}
 for r in policy['rules']:
  for rows in sorted({r['min_rows'],r['min_rows']+1,max(4096,r['min_rows']),16384}):
   for dtype in ('fp16','bf16'):add(r['dim'],rows,r['mode'],dtype,True)
  if r['min_rows']>1:
   for dtype in ('fp16','bf16'):add(r['dim'],r['min_rows']-1,r['mode'],dtype,False)
 for n,rows,mode in [(16,65,'hadamard_int4'),(16,4096,'hadamard_int4'),(32,17,'hadamard'),(64,65,'hadamard_int4'),(256,17,'hadamard')]:
  for dtype in ('fp16','bf16'):add(n,rows,mode,dtype,False)
 ordered=list(cases.values());random.Random(19090800+a.round).shuffle(ordered);report['case_order']=ordered;timing=json.loads((ROOT/'experiment_protocol.json').read_text())['timing']
 with torch.inference_mode():
  for ci,c in enumerate(ordered):
   dtype=torch.float16 if c['dtype']=='fp16' else torch.bfloat16;x=reference.make_input(torch,[c['rows'],c['dim']],dtype,'normal',2026,'cuda');fn=getattr(op,c['mode'])
   if c['mode']=='hadamard':auto=lambda:fn(x,1.,128,'auto')
   else:auto=lambda:fn(x,1.,128,'original','auto')
   funcs={'original128':lambda:fn(x,1.,128),'original256':lambda:fn(x,1.,256),'auto':auto};expected=funcs['original128']()
   for f in funcs.values():measure.exact(torch,f(),expected,'production auto')
   report['benchmarks'].append({**c,**measure.measure_graph(torch,funcs,a.round,ci,timing)});save();gc.collect()
 report['status']='PASS'
except Exception as e:report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());print(report['traceback'],flush=True)
report.update(finished_utc=measure.utc(),after=measure.snapshot());save();print(json.dumps({'status':report['status'],'cases':len(report['benchmarks']),'round':a.round}),flush=True)
