import json,subprocess,sys,hashlib
from pathlib import Path
root=Path('/data/infinitensor-2026')
src=root/'tools/ascend_div_probe'
report={'status':'PROBE_RUNNING','stages':[],'binaries':{}}
def stage(name,args):
    log=root/(name+'.log')
    with log.open('x') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
    report['stages'].append({'name':name,'args':args,'exit':r.returncode,'log':log.name,'sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
    print(name+' exit='+str(r.returncode),flush=True)
    (root/'div_probe_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return r.returncode
for label,enabled,mode in [('vector','OFF','vector'),('scalar','ON','both')]:
    build=root/('build/ascend_div_'+label)
    if stage('div_'+label+'_configure',['cmake','-S',str(src),'-B',str(build),'-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0','-DSOC_VERSION=Ascend910B1','-DRUN_MODE=npu','-DENABLE_SCALAR_DIV='+enabled,'-DCMAKE_BUILD_TYPE=Release']):continue
    if stage('div_'+label+'_build',['cmake','--build',str(build),'-j2']):continue
    exe=build/'ascend_div_probe'
    report['binaries'][label]=hashlib.sha256(exe.read_bytes()).hexdigest()
    stage('div_'+label+'_run',[str(exe),'--mode',mode])
report['status']='PROBE_FINISHED'
report['note']='Controller completion is not a numerical PASS; inspect each build/run exit and mismatch log.'
(root/'div_probe_report.json').write_text(json.dumps(report,indent=2)+'\n')
