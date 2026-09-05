import subprocess,json,sys
from pathlib import Path
root=Path('/data/infinitensor-2026')
repo=root/'Learning-CUDA';out=root/'runs/ascend_b7c5d82_first'
def step(name,args):
    log=out/(name+'.log')
    with log.open('x') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
    print(name+' exit='+str(p.returncode),flush=True)
    if p.returncode:sys.exit(p.returncode)
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip()
step('fetch',['git','-C',str(repo),'fetch','origin','feat/hadamard-cuda'])
target=subprocess.check_output(['git','-C',str(repo),'rev-parse','FETCH_HEAD']).decode().strip()
assert target=='b7c5d822befc92d3ca8063f7409071d9df1194b5',target
step('merge',['git','-C',str(repo),'merge','--ff-only','FETCH_HEAD'])
step('configure',['cmake','-S',str(repo/'03_hadamard_tc/a962695448-rgb/platforms/ascend'),'-B',str(out/'build'),'-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0','-DSOC_VERSION=Ascend910B1','-DRUN_MODE=npu','-DBUILD_VALIDATION=ON','-DCMAKE_BUILD_TYPE=Release'])
step('build',['cmake','--build',str(out/'build'),'-j2'])
print('ASCEND_INITIAL_BUILD_DONE '+target,flush=True)
