from pathlib import Path
import hashlib,json
root=Path('/data/infinitensor-2026') if Path('/data/infinitensor-2026').exists() else Path.home()/'infinitensor-2026';work=Path(__file__).resolve().parent
assert json.loads((work/'runs/holdout-suite.json').read_text())['status']=='PASS'
policy=json.loads((work/'runs/screen_policy.json').read_text());records={};sources={}
for i in (1,2,3):
    p=work/f'runs/holdout{i}/report.json';r=json.loads(p.read_text());assert r['status']=='PASS';sources[str(p.relative_to(work))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for c in r['benchmarks']:
        key=(c['dtype'],c['dim'],c['rows'],c['scale'],c['mode']);records.setdefault(key,[]).append(c['median_us'])
accepted=[];rejected=[]
for rule in policy['rules']:
    if rule['chosen'] is None:continue
    chosen=rule['chosen'];bad=[];gains=[];count=0
    for key,rounds in records.items():
        if key[1]!=rule['dim'] or key[4]!=rule['mode'] or key[2]<chosen['min_rows']:continue
        count+=1
        ratios=[t['packed'+str(chosen['block_threads'])]/min(t['original128'],t['original256']) for t in rounds];gains.extend(100*(1-x) for x in ratios)
        if max(ratios)>.95:bad.append({'case':key,'reduction_pct':[100*(1-x) for x in ratios]})
    item={**rule,'holdout_active_cases':count,'reduction_pct_range':[min(gains),max(gains)] if gains else None,'failures':bad}
    (rejected if bad or not count else accepted).append(item)
out={'gpu':policy['gpu'],'status':'PASS','accepted_rules':accepted,'rejected_rules':rejected,'holdout_source_hashes':sources,'screen_policy_sha256':hashlib.sha256((work/'runs/screen_policy.json').read_bytes()).hexdigest()}
target=work/'runs/policy_validation.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
