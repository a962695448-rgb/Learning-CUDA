import subprocess,sys
from pathlib import Path
root=Path('/data/infinitensor-2026');repo=root/'Learning-CUDA'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip()
subprocess.run(['git','-C',str(repo),'fetch','origin','feat/hadamard-cuda'],check=True)
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','FETCH_HEAD']).decode().strip()
assert head=='1b305d3e3a07b28b6879596babf47aadbf84ba0c',head
subprocess.run(['git','-C',str(repo),'merge','--ff-only','FETCH_HEAD'],check=True)
p=repo/'03_hadamard_tc/a962695448-rgb'
subprocess.run([sys.executable,str(p/'platforms/ascend/run_platform.py'),'--cann-root','/usr/local/Ascend/cann-9.0.0','--quick','--pilot-benchmark','--build-jobs','2','--output',str(root/'runs/ascend_quick_1b305')],cwd=p,check=True)
