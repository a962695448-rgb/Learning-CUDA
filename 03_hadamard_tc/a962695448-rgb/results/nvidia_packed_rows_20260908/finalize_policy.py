from pathlib import Path
import hashlib,json
root=Path('/data/infinitensor-2026') if Path('/data/infinitensor-2026').exists() else Path.home()/'infinitensor-2026';work=root/'profile-dispatch-20260908'
assert json.loads((work/'runs/refined-suite.json').read_text())['status']=='PASS'
old=json.loads((work/'runs/policy_validation.json').read_text());refine=json.loads((work/'refinement_protocol.json').read_text());records={};sources={}
for i in (1,2,3):
    p=work/f'runs/refined{i}/report.json';r=json.loads(p.read_text());assert r['status']=='PASS';sources[str(p.relative_to(work))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for c in r['benchmarks']:
        key=(c['dtype'],c['dim'],c['rows'],c['scale'],c['mode']);records.setdefault(key,[]).append(c['median_us'])
accepted=[];rejected=[]
for rule in refine['rules']:
    gains=[];bad=[];count=0
    for key,rounds in records.items():
        if key[1]!=rule['dim'] or key[4]!=rule['mode'] or key[2]<rule['min_rows']:continue
        count+=1;values=[100*(1-t['packed256']/min(t['original128'],t['original256'])) for t in rounds];gains.extend(values)
        if min(values)<5:bad.append({'case':key,'reduction_pct':values})
    item={**rule,'active_cases':count,'reduction_pct_range':[min(gains),max(gains)],'failures':bad}
    (rejected if bad else accepted).append(item)
rules=[{'dim':r['dim'],'mode':r['mode'],'min_rows':r['chosen']['min_rows'],'block_threads':r['chosen']['block_threads'],'validation':'first_holdout','reduction_pct_range':r['reduction_pct_range']} for r in old['accepted_rules']]
rules += [{k:r[k] for k in ('dim','mode','min_rows','block_threads','reduction_pct_range')}|{'validation':'independent_refinement'} for r in accepted]
out={'status':'ANALYZED','gpu':old['gpu'],'rules':sorted(rules,key=lambda r:(r['dim'],r['mode'])),'refinement_accepted':accepted,'refinement_rejected':rejected,'original_rejected_rules_preserved':len(old['rejected_rules']),'refined_source_hashes':sources,'first_policy_sha256':hashlib.sha256((work/'runs/policy_validation.json').read_bytes()).hexdigest(),'refinement_protocol_sha256':hashlib.sha256((work/'refinement_protocol.json').read_bytes()).hexdigest()}
p=work/'runs/final_policy.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'gpu':out['gpu'],'rules':out['rules'],'refined_rejected':len(rejected)},indent=2))
