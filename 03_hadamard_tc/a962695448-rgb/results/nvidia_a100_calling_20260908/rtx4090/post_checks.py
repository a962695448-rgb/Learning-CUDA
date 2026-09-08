from pathlib import Path
import json,os,subprocess,sys,time,traceback
w=Path(__file__).resolve().parent
for _ in range(450):
 p=w/'runs/suite.json'
 if p.exists():
  try:state=json.loads(p.read_text())['status']
  except json.JSONDecodeError:state='RUNNING'
  if state=='PASS':break
  if state=='FAIL':raise RuntimeError('Final suite failed; no profiling')
 time.sleep(2)
else:raise TimeoutError('Final suite did not complete')
out=w/'checks';out.mkdir(exist_ok=False);r={'status':'RUNNING','steps':[],'scope':'Safety and execution-path checks; profiler times are not benchmark samples.'};env={k:os.environ[k] for k in ('HOME','USER','LOGNAME','LANG','PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES') if k in os.environ};env['PATH']='/usr/local/cuda/bin:'+env.get('PATH','')
def run(name,cmd,timeout=180,required=True):
 with (out/(name+'.log')).open('xb') as f:p=subprocess.run(list(map(str,cmd)),env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 r['steps'].append({'name':name,'argv':list(map(str,cmd)),'exit_code':p.returncode});(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 if required:assert p.returncode==0,name
 return (out/(name+'.log')).read_text(errors='replace')
try:
 gpu=json.loads((w/'launch.json').read_text())['gpu'];is_a100=gpu=='NVIDIA A100-SXM4-40GB';r['gpu']=gpu
 binary=w/'production/project/build/hadamard'
 if is_a100:
  test=out/'memory_check';run('build_sanitizer',['nvcc','-std=c++17','-O2','-arch=sm_80','-lineinfo','-I'+str(w/'production/project/include'),w/'memory_check.cu','-o',test])
  run('native_safety',[test])
  for tool in ('memcheck','synccheck'):
   text=run(tool,['compute-sanitizer','--tool',tool,'--error-exitcode','97',test]);assert 'ERROR SUMMARY: 0 errors' in text
 cases=[('active',2,256),('fallback',2,255)] if is_a100 else [('active',8,4096),('fallback',16,65)]
 for name,n,rows in cases:
  run(name+'_profile',['nsys','profile','--sample=none','--cpuctxsw=none','--trace=cuda','--force-overwrite=true','-o',out/name,binary,'--benchmark','--batch','1','--seq',str(rows),'--heads','1','--dim',str(n),'--row-layout','auto','--warmup','1','--repetitions','2'])
  text=run(name+'_stats',['nsys','stats','--report','cuda_gpu_kern_sum,cuda_api_sum','--format','csv',out/(name+'.nsys-rep')]);assert ('packed_rows_kernel' in text)==(name=='active')
 text=run('ncu',['ncu','--set','basic','--kernel-name','regex:packed_rows_kernel','--launch-count','1','--csv',binary,'--benchmark','--batch','1','--seq','4096','--heads','1','--dim','8','--row-layout','auto','--warmup','1','--repetitions','1'],required=False)
 r['ncu_status']='PERMISSION_DENIED' if 'ERR_NVGPUCTRPERM' in text else ('COLLECTED' if r['steps'][-1]['exit_code']==0 else 'OTHER_FAILURE')
 r['status']='PASS'
except Exception as e:r.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
