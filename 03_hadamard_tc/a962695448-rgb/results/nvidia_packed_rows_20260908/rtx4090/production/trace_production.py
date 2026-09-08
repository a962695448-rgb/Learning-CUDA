from pathlib import Path
import json,os,subprocess
root=Path('/data/infinitensor-2026') if Path('/data/infinitensor-2026').exists() else Path.home()/'infinitensor-2026';work=root/'profile-dispatch-20260908/production'
assert json.loads((work/'runs/performance-suite.json').read_text())['status']=='PASS'
out=work/'runs/nsys';assert not out.exists();out.mkdir()
env={k:os.environ[k] for k in ('HOME','USER','LOGNAME','LANG','PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES') if k in os.environ};env['PATH']=env.get('PATH','')+':/usr/local/cuda/bin';binary=work/'project/build/hadamard';report={'status':'RUNNING','steps':[],'scope':'Profiler traces only; timings collected here are not performance benchmark samples. Child environment is allowlisted.'}
try:
    for label,n,rows in [('packed',8,4096),('fallback',16,65)]:
        cmd=['nsys','profile','--sample=none','--cpuctxsw=none','--trace=cuda','--force-overwrite=true','-o',str(out/label),str(binary),'--benchmark','--batch','1','--seq',str(rows),'--heads','1','--dim',str(n),'--row-layout','auto','--warmup','1','--repetitions','2']
        with (out/(label+'.log')).open('xb') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        report['steps'].append({'argv':cmd,'exit_code':r.returncode});assert r.returncode==0
        cmd=['nsys','stats','--report','cuda_gpu_kern_sum,cuda_api_sum','--format','csv',str(out/(label+'.nsys-rep'))]
        with (out/(label+'-stats.log')).open('xb') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        report['steps'].append({'argv':cmd,'exit_code':r.returncode});assert r.returncode==0
        text=(out/(label+'-stats.log')).read_text(errors='replace')
        assert ('packed_rows_kernel' in text)==(label=='packed')
        assert 'warp_kernel' in text
    report['status']='PASS'
except Exception as e:report.update(status='FAIL',error=repr(e))
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
