from pathlib import Path
import hashlib,json,os,subprocess,sys,time
w=Path(__file__).resolve().parent;root=Path.home()/'infinitensor-2026';report={'status':'RUNNING','steps':[]};target=w/'runs/calibration-suite.json';target.parent.mkdir(exist_ok=True);assert not target.exists()
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def run(name,args):
 cmd=[sys.executable,*map(str,args)];record={'name':name,'argv':cmd};report['steps'].append(record);save()
 with (w/'runs'/(name+'.log')).open('xb') as log:
  p=subprocess.Popen(cmd,cwd=w,stdout=log,stderr=subprocess.STDOUT);record['pid']=p.pid;save();record['exit_code']=p.wait()
 save();assert record['exit_code']==0,name
try:
 assert json.loads((w.parent/'eager-suite.json').read_text())['status']=='PASS'
 run('correctness',[w/'run_packed_rows.py','--stage','correctness','--out',w/'runs/correctness','--reference-repo',root/'hadamard-cuda-a100/fast-hadamard-transform'])
 assert json.loads((w/'runs/correctness/report.json').read_text())['status']=='PASS'
 run('screen',[w/'run_three.py','screen']);assert json.loads((w/'runs/screen-suite.json').read_text())['status']=='PASS'
 run('freeze',[w/'freeze_policy.py']);assert json.loads((w/'runs/screen_policy.json').read_text())['status']=='FROZEN_BEFORE_HOLDOUT'
 run('holdout',[w/'run_three.py','holdout']);assert json.loads((w/'runs/holdout-suite.json').read_text())['status']=='PASS'
 run('validate_policy',[w/'check_policy.py']);report['status']='PASS'
except Exception as e:report.update(status='FAIL',error=repr(e))
save();print(json.dumps(report),flush=True)
