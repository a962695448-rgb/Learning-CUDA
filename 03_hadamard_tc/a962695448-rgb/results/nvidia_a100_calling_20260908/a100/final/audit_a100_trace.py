from pathlib import Path
import csv,hashlib,json,os,subprocess
w=Path.home()/'infinitensor-2026/a100-eager-20260908/final';out=w/'checks';old=json.loads((out/'report.json').read_text());assert old['status']=='FAIL'
assert all(s['exit_code']==0 for s in old['steps']) and old['steps'][-1]['name']=='fallback_stats'
def names(path):
 return [row[-1] for row in csv.reader(path.read_text(errors='replace').splitlines()) if row and 'hadamard::' in row[-1]]
active=names(out/'active_stats.log');fallback=names(out/'fallback_stats.log')
def has(values,kernel,quant):return any(kernel+'<__half, (int)2, (bool)1, (bool)'+str(quant)+'>' in x for x in values)
assert has(active,'packed_rows_kernel',0)
assert not has(fallback,'packed_rows_kernel',0)
assert has(fallback,'warp_kernel',0)
assert has(fallback,'packed_rows_kernel',1)
report={'status':'PASS','scope':'Audit of existing raw traces without rerunning profiling. N2/M255 falls back only for transform; fused N2 remains packed because its independent threshold is M>=1.','initial_wrapper_status':'FAIL','initial_failure_reason':'Wrapper incorrectly required absence of every packed kernel in a multi-path CLI benchmark. Correct per-operation dispatch is verified from template instantiations. No computation, thresholds, or original logs changed.','active_kernel_names':active,'fallback_kernel_names':fallback,'source_hashes':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('report.json','active_stats.log','fallback_stats.log')}}
target=out/'trace-scope-audit.json';assert not target.exists();target.write_text(json.dumps(report,indent=2)+'\n')
log=out/'ncu-resumed.log';assert not log.exists();cmd=['/usr/local/cuda/bin/ncu','--set','basic','--kernel-name','regex:packed_rows_kernel','--launch-count','1','--csv',str(w/'production/project/build/hadamard'),'--benchmark','--batch','1','--seq','4096','--heads','1','--dim','8','--row-layout','auto','--warmup','1','--repetitions','1']
env={k:os.environ[k] for k in ('HOME','USER','LOGNAME','LANG','PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES') if k in os.environ}
with log.open('xb') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
text=log.read_text(errors='replace');state='PERMISSION_DENIED' if 'ERR_NVGPUCTRPERM' in text else ('COLLECTED' if p.returncode==0 and 'Metric Name' in text else 'NO_METRICS_OR_OTHER_FAILURE')
r={'status':state,'argv':cmd,'exit_code':p.returncode,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()};(out/'ncu-resumed.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'trace_scope_audit':'PASS','ncu':r}))
