import json,subprocess,sys,hashlib
from pathlib import Path
root=Path('/data/infinitensor-2026')
exe=root/'build/ascend_stage2/ascend_stage2'
stages=[('stage2_pad_one',['--stage','pad','--dtype','fp16','--n','1']),('stage2_pad_full',['--stage','pad','--dtype','both']),('stage2_rne_full',['--stage','rne','--dtype','both'])]
report={'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'stages':[]}
code=0
for name,args in stages:
    log=root/(name+'.log')
    with log.open('x') as f:r=subprocess.run([str(exe),*args],stdout=f,stderr=subprocess.STDOUT)
    report['stages'].append({'name':name,'args':args,'exit':r.returncode,'log':log.name,'sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
    print(name+' exit='+str(r.returncode),flush=True)
    if r.returncode:code=r.returncode;break
report['status']='PASS' if code==0 else 'FAIL'
(root/'stage2_checks_report.json').write_text(json.dumps(report,indent=2)+'\n')
sys.exit(code)
