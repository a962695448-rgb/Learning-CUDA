from pathlib import Path
import json,os,subprocess,sys,time,traceback
root=Path('/data/infinitensor-2026') if Path('/data/infinitensor-2026').exists() else Path.home()/'infinitensor-2026';work=root/'profile-dispatch-20260908';stage=sys.argv[1];assert stage in ('screen','holdout')
assert json.loads((work/'runs/correctness/report.json').read_text())['status']=='PASS'
target=work/'runs'/(stage+'-suite.json');assert not target.exists()
state={'status':'RUNNING','pid':os.getpid(),'stage':stage,'runs':[]}
def save():target.write_text(json.dumps(state,indent=2)+'\n')
save();py=root/'.venv/bin/python' if str(root).startswith('/data') else root/'hadamard-cuda-a100/.venv/bin/python';ref=root/'fast-hadamard-transform' if str(root).startswith('/data') else root/'hadamard-cuda-a100/fast-hadamard-transform'
env=dict(os.environ,PATH=str(py.parent)+':/usr/local/cuda/bin:'+os.environ.get('PATH',''),CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='8.9' if str(root).startswith('/data') else '8.0',MAX_JOBS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONOPTIMIZE='0',PYTHONUNBUFFERED='1')
try:
    for index in (1,2,3):
        out=work/'runs'/f'{stage}{index}';assert not out.exists()
        idle=0
        for _ in range(30):
            values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).split();idle=idle+1 if values and all(int(v)==0 for v in values) else 0
            if idle>=2:break
            time.sleep(2)
        assert idle>=2,'GPU did not become idle'
        argv=[str(py),str(work/'run_packed_rows.py'),'--stage',stage,'--round',str(index),'--out',str(out),'--reference-repo',str(ref)]
        record={'index':index,'argv':argv};state['runs'].append(record)
        with (work/'runs'/(out.name+'.controller.log')).open('xb') as log:
            p=subprocess.Popen(argv,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);record['pid']=p.pid;save();record['exit_code']=p.wait()
        save();assert record['exit_code']==0
        assert json.loads((out/'report.json').read_text())['status']=='PASS'
    state['status']='PASS'
except Exception as e:state.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
save();print(json.dumps(state),flush=True)
