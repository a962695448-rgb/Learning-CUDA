"""Prospective full-matrix confirmation on exactly the previously checked binaries."""
import argparse
import base64
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--base-run',type=Path,required=True)
args=parser.parse_args();BASE=args.base_run.resolve()
OUT=ROOT/'results-remote';OUT.mkdir(exist_ok=False)
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
manifest=json.loads((ROOT/'INPUT_MANIFEST.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
expected=json.loads((ROOT/'BASE_EXPECTATIONS.json').read_text())
assert hashlib.sha256((BASE/'INPUT_MANIFEST.json').read_bytes()).hexdigest()==expected['input_manifest_sha256']


def check_base():
    for name,sha in expected['input_manifest'].items():assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==sha,name
    for label,folder,name in [('control','control_a','ip_control_a'),('twin','control_b','ip_control_b'),('candidate','candidate','ip_candidate')]:
        path=BASE/'build'/folder/(name+'.so')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['binaries'][label],label


check_base()
import numpy as np
import torch
assert torch.__version__==expected['torch'] and torch.version.cuda==expected['torch_cuda']
assert np.__version__==expected['numpy']
assert list(torch.cuda.get_device_capability())==expected['capability']
summary={'status':'RUNNING','jobs':[],'same_tested_binaries':True,'base_input_files':len(expected['input_manifest']),
  'base_binaries':expected['binaries'],'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
  'gpu':torch.cuda.get_device_name(),'python':sys.version,'capability':list(torch.cuda.get_device_capability()),
  'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),
  'nvidia_smi_before':subprocess.check_output(['nvidia-smi'],text=True)}


def save():
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


save()
try:
    for round_id in range(3):
        start=time.monotonic()
        with (OUT/f'round-{round_id}.log').open('x') as log:
            result=subprocess.run([sys.executable,'-u',str(ROOT/'bench.py'),'--round',str(round_id),'--base-run',str(BASE)],
              cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        summary['jobs'].append({'round':round_id,'returncode':result.returncode,'seconds':time.monotonic()-start});save()
        assert result.returncode==0,round_id
        print('ROUND_COMPLETE',round_id,flush=True)
    reports=[json.loads((OUT/f'round-{i}.json').read_text()) for i in range(3)]
    combined={'status':'ACCEPT' if all(r['status']=='ACCEPT' for r in reports) else 'REJECT',
      'protocol':json.loads((ROOT/'PROTOCOL.json').read_text())['cuda']}
    for key in ('records','host_records','groups','calibration'):combined[key]=[x for r in reports for x in r[key]]
    (OUT/'benchmark.json').write_text(json.dumps(combined,indent=2)+'\n')
    summary.update(status='PASS',candidate_decision=combined['status'])
except Exception:
    summary.update(status='FAIL',error=traceback.format_exc());print(summary['error'],flush=True)
check_base()
for name,sha in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
summary['all_inputs_and_binaries_unchanged']=True
summary['nvidia_smi_after']=subprocess.check_output(['nvidia-smi'],text=True);save()
files={str(p.relative_to(ROOT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()} for p in OUT.iterdir() if p.is_file()}
raw=json.dumps(files,ensure_ascii=True).encode();gz=gzip.compress(raw,mtime=0);encoded=base64.b64encode(gz).decode()
(ROOT/'FINAL_EXPORT.base64.txt').write_text(encoded);parts=[]
for i,start in enumerate(range(0,len(encoded),400000)):
    name=f'FINAL_EXPORT_PART_{i:02d}.txt';part=encoded[start:start+400000];(ROOT/name).write_text(part)
    parts.append({'name':name,'characters':len(part),'sha256':hashlib.sha256(part.encode()).hexdigest()})
receipt={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'gzip_bytes':len(gz),'gzip_sha256':hashlib.sha256(gz).hexdigest(),
  'base64_characters':len(encoded),'parts':parts,'status':summary['status'],'decision':summary.get('candidate_decision')}
(ROOT/'FINAL_EXPORT_RECEIPT.txt').write_text(json.dumps(receipt));print(receipt,flush=True)
