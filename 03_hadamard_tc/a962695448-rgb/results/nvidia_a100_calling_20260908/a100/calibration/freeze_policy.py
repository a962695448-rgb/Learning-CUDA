from pathlib import Path
import hashlib,json,math
root=Path('/data/infinitensor-2026') if Path('/data/infinitensor-2026').exists() else Path.home()/'infinitensor-2026';work=Path(__file__).resolve().parent;target=work/'runs/screen_policy.json';assert not target.exists()
assert json.loads((work/'runs/screen-suite.json').read_text())['status']=='PASS'
cases={};sources={};gpu=None
for i in (1,2,3):
    p=work/f'runs/screen{i}/report.json';r=json.loads(p.read_text());assert r['status']=='PASS';gpu=r['environment']['gpu'];sources[str(p.relative_to(work))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for c in r['benchmarks']:
        key=(c['dtype'],c['dim'],c['rows'],c['scale'],c['mode']);cases.setdefault(key,[]).append(c['median_us'])
rules=[]
for n in (1,2,4,8,16):
    for mode in ('hadamard','hadamard_int4'):
        candidates=[]
        for threshold in (1,16,64,256,4096,16384):
            active=[times for k,times in cases.items() if k[1]==n and k[4]==mode and k[2]>=threshold]
            for threads in (128,256):
                ratios=[t['packed'+str(threads)]/min(t['original128'],t['original256']) for times in active for t in times]
                if ratios and max(ratios)<=.95:candidates.append({'min_rows':threshold,'block_threads':threads,'geomean_ratio':math.exp(sum(map(math.log,ratios))/len(ratios)),'worst_reduction_pct':100*(1-max(ratios))})
        chosen=min(candidates,key=lambda c:(c['min_rows'],c['geomean_ratio'])) if candidates else None
        rules.append({'dim':n,'mode':mode,'chosen':chosen})
report={'status':'FROZEN_BEFORE_HOLDOUT','gpu':gpu,'source_hashes':sources,'rules':rules,'selection':'Among candidates passing >=5% in every activated screening case and each of three rounds, choose widest row range, then lowest geometric mean ratio. Baseline is better same-run original128/256.','holdout_policy':'A rule is rejected as a whole if any activated holdout case loses its >=5% three-round gate; do not refit thresholds using the same holdout.'}
target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
