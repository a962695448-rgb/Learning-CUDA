import subprocess,sys
from pathlib import Path
root=Path('/data/infinitensor-2026');repo=root/'Learning-CUDA'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip()
subprocess.run(['git','-C',str(repo),'fetch','origin','feat/hadamard-cuda'],check=True)
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','FETCH_HEAD']).decode().strip()
assert head=='f614762a69e75524db65b47fbf7d6d01836db438',head
subprocess.run(['git','-C',str(repo),'merge','--ff-only','FETCH_HEAD'],check=True)
project=repo/'03_hadamard_tc/a962695448-rgb'
for label,extra in [('off',[]),('on',['--vector-scale'])]:
    out=root/('runs/ascend_scale_'+label+'_f614762')
    args=[sys.executable,str(project/'platforms/ascend/run_platform.py'),'--cann-root','/usr/local/Ascend/cann-9.0.0','--build-jobs','2','--pilot-benchmark','--output',str(out)]+extra
    print('START_VARIANT '+label,flush=True)
    result=subprocess.run(args,cwd=project)
    print('END_VARIANT '+label+' exit='+str(result.returncode),flush=True)
    if result.returncode:sys.exit(result.returncode)
