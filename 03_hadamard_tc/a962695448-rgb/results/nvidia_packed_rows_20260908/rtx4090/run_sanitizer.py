from pathlib import Path
import hashlib,json,os,subprocess
root=Path('/data/infinitensor-2026') if Path('/data/infinitensor-2026').exists() else Path.home()/'infinitensor-2026';work=root/'profile-dispatch-20260908';out=work/'runs/sanitizer'
assert json.loads((work/'runs/holdout-suite.json').read_text())['status']=='PASS'
assert not out.exists();out.mkdir()
manifest=json.loads((work/'experiment_manifest.json').read_text())
for name,item in manifest['files'].items():assert hashlib.sha256((work/name).read_bytes()).hexdigest()==item['sha256']
env=dict(os.environ,PATH=os.environ.get('PATH','')+':/usr/local/cuda/bin');arch='89' if str(root).startswith('/data') else '80'
cmds=[['nvcc','-O3','-std=c++17','-lineinfo','-arch=sm_'+arch,'-I'+str(work),'-I'+str(work/'project/include'),str(work/'memory_check.cu'),'-o',str(out/'memory_check')],[str(out/'memory_check')],['compute-sanitizer','--tool','memcheck','--error-exitcode','9',str(out/'memory_check')],['compute-sanitizer','--tool','synccheck','--error-exitcode','9',str(out/'memory_check')]]
report={'status':'RUNNING','source_sha256':hashlib.sha256((work/'memory_check.cu').read_bytes()).hexdigest(),'steps':[]}
try:
    for i,cmd in enumerate(cmds):
        with (out/f'step{i}.log').open('xb') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        report['steps'].append({'argv':cmd,'exit_code':r.returncode});assert r.returncode==0
        if i>0:assert 'MEMORY_FIXTURES_PASS cases=160' in (out/f'step{i}.log').read_text()
        if i>1:assert 'ERROR SUMMARY: 0 errors' in (out/f'step{i}.log').read_text()
    report['status']='PASS'
except Exception as e:report.update(status='FAIL',error=repr(e))
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
