from pathlib import Path
import json,subprocess,sys
w=Path(__file__).resolve().parent;(w/'runs').mkdir(exist_ok=False);out=w/'runs/suite.json';report={'status':'RUNNING','steps':[],'scope':'Execution/correctness status; performance acceptance is evaluated separately from raw results.'}
def save():out.write_text(json.dumps(report,indent=2)+'\n')
def run(name,args,report_path):
 record={'name':name,'argv':[sys.executable,*map(str,args)]};report['steps'].append(record);save()
 with (w/'runs'/(name+'.log')).open('xb') as log:
  p=subprocess.Popen(record['argv'],cwd=w,stdout=log,stderr=subprocess.STDOUT);record['pid']=p.pid;save();record['exit_code']=p.wait()
 save();assert record['exit_code']==0 and json.loads(report_path.read_text())['status']=='PASS',name
try:
 run('validation',[w/'production/production_validate.py'],w/'production/runs/validation/report.json')
 for n in (1,2,3):run('graph'+str(n),[w/'final_perf.py','--round',str(n)],w/f'runs/graph{n}.json')
 for n in (1,2,3):run('eager'+str(n),[w/'eager_ablation.py','--round',str(n)],w/f'eager/round{n}.json')
 report['status']='PASS'
except Exception as e:report.update(status='FAIL',error=repr(e))
save();print(json.dumps(report),flush=True)
