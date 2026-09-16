python3 - <<'PY'
from pathlib import Path
import hashlib,json,shutil,subprocess,os
root=Path('/data/api-optimization-20260917');upload=Path('/data')
for name,sha in {'run_gpu.py':'1c6cb74ea2619882d0b67e7df76afb2115c60d9040b02a8f44ab8ad47bba82a2','verify_out_buffers.py':'07daf545e33a099e530586526c19afb79c80efdf2d78273150d1b659312a5cd8'}.items():assert hashlib.sha256((upload/name).read_bytes()).hexdigest()==sha
s=json.loads((root/'results/summary.json').read_text());assert s['status']=='FAIL' and all(j['returncode']==0 for j in s['jobs'])
target=root/'candidate/cuda/03_hadamard_tc/a962695448-rgb/scripts/verify_out_buffers.py'
for old,new in [(root/'run_gpu.py',root/'run_gpu_initial.py'),(target,root/'verify_out_buffers_initial.py'),(root/'INPUT_MANIFEST.json',root/'INPUT_MANIFEST.initial.json')]:
 assert not new.exists();shutil.copy2(old,new)
shutil.copy2(upload/'run_gpu.py',root/'run_gpu.py');shutil.copy2(upload/'verify_out_buffers.py',target)
m=json.loads((root/'INPUT_MANIFEST.json').read_text());m[str(target.relative_to(root))]=hashlib.sha256(target.read_bytes()).hexdigest();(root/'INPUT_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
env=dict(os.environ,API_OPT_RESULTS_DIR='results-retry',API_OPT_CUDA_ONLY='1');log=(root/'run-retry.log').open('x');p=subprocess.Popen(['python3',str(root/'run_gpu.py')],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);print('RETRY RUNNING',p.pid)
PY
