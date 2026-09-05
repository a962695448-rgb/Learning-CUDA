import subprocess,json,time,hashlib,os
from pathlib import Path
r=Path('/data/infinitensor-2026/runs/biren-api-full-2cbaf41')
b=r/'validate_and_benchmark'
s=json.loads((r/'run_summary.json').read_text());env=dict(os.environ);env.update(s['sdk_environment'])
sdk=Path(env['BIREN_HOME']);supa=Path(env['SUPA_PATH']);brcc=sdk/'brcc'
env['PATH']=os.pathsep.join([str(brcc/'bin'),str(supa/'bin'),os.environ.get('PATH','')])
libs=[str(p) for p in [supa/'lib',brcc/'lib'] if p.is_dir()];old=os.environ.get('LD_LIBRARY_PATH','');env['LD_LIBRARY_PATH']=os.pathsep.join(libs+([old] if old else []))
records=[]
for i in [2,3]:
 rec={'run':i,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'binary_sha256':hashlib.sha256(b.read_bytes()).hexdigest(),'sdk_environment':s['sdk_environment']}
 with (r/('brsmi_run%d_before.log'%i)).open('x') as f:rec['brsmi_before_exit']=subprocess.run(['brsmi'],stdout=f,stderr=subprocess.STDOUT,env=env).returncode
 args=[str(b),'--benchmark','--csv',str(r/('benchmark_run%d.csv'%i)),'--groups','5','--repeats','100'];rec['args']=args
 with (r/('benchmark_run%d.log'%i)).open('x') as f:rec['exit']=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
 with (r/('brsmi_run%d_after.log'%i)).open('x') as f:rec['brsmi_after_exit']=subprocess.run(['brsmi'],stdout=f,stderr=subprocess.STDOUT,env=env).returncode
 rec['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());records.append(rec);(r/'repeat_runs.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(rec),flush=True)
 if rec['exit']:raise SystemExit(rec['exit'])
