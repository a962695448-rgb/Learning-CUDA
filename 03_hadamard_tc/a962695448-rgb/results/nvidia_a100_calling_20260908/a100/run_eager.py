from pathlib import Path
import json,os,subprocess,sys,time
root=Path(__file__).resolve().parent;report={'status':'RUNNING','runs':[]};out=root/'eager-suite.json';assert not out.exists()
for _ in range(360):
 p=root/'production_retry/runs/validation/report.json'
 if p.exists():
  try:status=json.loads(p.read_text())['status']
  except json.JSONDecodeError:status='RUNNING'
  if status=='PASS':break
  if status=='FAIL':raise RuntimeError('Baseline validation failed')
 time.sleep(2)
else:raise TimeoutError('Baseline not complete')
try:
 for n in (1,2,3):
  with (root/f'eager-round{n}.log').open('xb') as f:
   p=subprocess.Popen([sys.executable,str(root/'eager_ablation.py'),'--round',str(n)],cwd=root,stdout=f,stderr=subprocess.STDOUT)
   report['runs'].append({'round':n,'pid':p.pid});out.write_text(json.dumps(report,indent=2)+'\n');report['runs'][-1]['exit_code']=p.wait()
  assert report['runs'][-1]['exit_code']==0
  assert json.loads((root/f'eager/round{n}.json').read_text())['status']=='PASS'
 report['status']='PASS'
except Exception as e:report.update(status='FAIL',error=repr(e))
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
