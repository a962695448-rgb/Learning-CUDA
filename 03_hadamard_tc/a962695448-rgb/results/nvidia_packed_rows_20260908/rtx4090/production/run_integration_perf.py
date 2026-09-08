from pathlib import Path
import json,os,subprocess,time
root=Path('/data/infinitensor-2026') if Path('/data/infinitensor-2026').exists() else Path.home()/'infinitensor-2026';work=root/'profile-dispatch-20260908/production'
assert json.loads((work/'runs/validation/report.json').read_text())['status']=='PASS'
py=root/'.venv/bin/python' if str(root).startswith('/data') else root/'hadamard-cuda-a100/.venv/bin/python'
env=dict(os.environ,PATH=str(py.parent)+':/usr/local/cuda/bin:'+os.environ.get('PATH',''),CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='8.9' if str(root).startswith('/data') else '8.0',MAX_JOBS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
state={'status':'RUNNING','runs':[]};target=work/'runs/performance-suite.json';assert not target.exists()
try:
    for i in (1,2,3):
        with (work/f'runs/performance{i}.log').open('xb') as log:
            p=subprocess.Popen([str(py),str(work/'integration_perf.py'),'--round',str(i)],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);entry={'round':i,'pid':p.pid};state['runs'].append(entry);target.write_text(json.dumps(state));entry['exit_code']=p.wait()
        assert entry['exit_code']==0 and json.loads((work/f'runs/performance{i}.json').read_text())['status']=='PASS'
    state['status']='PASS'
except Exception as e:state.update(status='FAIL',error=repr(e))
target.write_text(json.dumps(state,indent=2)+'\n')
