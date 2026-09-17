"""Run the preregistered supplemental compatibility check before timing."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--base-run',type=Path,required=True)
args=parser.parse_args();BASE=args.base_run.resolve()
expected=json.loads((ROOT/'BASE_EXPECTATIONS.json').read_text())
for name,sha in expected['input_manifest'].items():assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==sha,name
for label,folder,name in [('control','control_a','ip_control_a'),('twin','control_b','ip_control_b'),('candidate','candidate','ip_candidate')]:
    assert hashlib.sha256((BASE/'build'/folder/(name+'.so')).read_bytes()).hexdigest()==expected['binaries'][label],label
import numpy as np
import torch
assert torch.__version__==expected['torch'] and torch.version.cuda==expected['torch_cuda']
assert np.__version__==expected['numpy'] and list(torch.cuda.get_device_capability())==expected['capability']
precheck=ROOT/'precheck';precheck.mkdir(exist_ok=False)
(precheck/'candidate').symlink_to(BASE/'candidate',target_is_directory=True)
(precheck/'build').symlink_to(BASE/'build',target_is_directory=True)
(precheck/'results-remote').mkdir()
environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
script=BASE/'tools/verify_overflow_pairs.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()==expected['input_manifest']['tools/verify_overflow_pairs.py']
with (ROOT/'precheck.log').open('x') as log:
    result=subprocess.run([sys.executable,'-u',str(script),str(precheck)],cwd=ROOT,env=environment,
                          stdout=log,stderr=subprocess.STDOUT,timeout=180)
assert result.returncode==0,'Supplemental compatibility check failed'
report=json.loads((precheck/'results-remote/overflow-compatibility.json').read_text())
assert report['status']=='PASS' and len(report['records'])==512
(ROOT/'PRECHECK.json').write_text(json.dumps({'status':'PASS','cases':512,
    'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'timing_script_unchanged':True},indent=2)+'\n')
print('PREREGISTERED_SUPPLEMENTAL_CHECK_PASS 512',flush=True)
result=subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),'--base-run',str(BASE)],cwd=ROOT,env=environment)
(ROOT/'ORCHESTRATION_END.json').write_text(json.dumps({'returncode':result.returncode}))
raise SystemExit(result.returncode)
