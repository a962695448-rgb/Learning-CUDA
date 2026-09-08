from pathlib import Path
import csv,hashlib,json,math,statistics,sys
root=Path(sys.argv[1]);out=root/'production_analysis.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
validation=json.loads((root/'production/runs/validation/report.json').read_text());assert validation['status']=='PASS'
policy=json.loads((root/'runs/final_policy.json').read_text())
rows=[];eager=[];keys=set();binaries=set()
for round in (1,2,3):
 r=json.loads((root/f'production/runs/performance{round}.json').read_text());assert r['status']=='PASS' and r['round']==round
 assert r['validation_sha256']==sha(root/'production/runs/validation/report.json')
 assert r['policy_sha256']==sha(root/'runs/final_policy.json')
 binaries.add(r['binary_sha256']);assert len(r['benchmarks'])==72 and len(r['eager'])==8
 runkeys=set()
 for c in r['benchmarks']:
  key=tuple(c[k] for k in ('mode','dim','rows','dtype','scale'));assert key not in runkeys;runkeys.add(key)
  assert c['captured_calls_per_graph']==64 and c['replays_per_group']==20 and c['graph_warmup_replays']==256
  assert c['independent_output_buffers'] and c['cross_method_output_pointers_disjoint'] and c['outputs_bitwise_equal_eager_before_and_after']
  for method,v in c['raw_event_intervals_ms'].items():
   assert len(v)==5 and all(x>0 and math.isfinite(x) for x in v)
   calculated=[x*1000/1280 for x in v]
   assert all(math.isclose(a,b,rel_tol=1e-12) for a,b in zip(calculated,c['samples_us'][method]))
   assert math.isclose(statistics.median(calculated),c['median_us'][method],rel_tol=1e-12)
   assert math.isclose(c['median_us'][method]/1000,c['median_ms'][method],rel_tol=1e-12)
  selected=any(x['dim']==c['dim'] and x['mode']==c['mode'] and c['rows']>=x['min_rows'] for x in policy['rules'])
  assert selected==c['activated']
  best=min(c['median_us']['original128'],c['median_us']['original256'])
  base=best if selected else c['median_us']['original128'];auto=c['median_us']['auto']
  rows.append({**{k:c[k] for k in ('mode','dim','rows','dtype','scale','activated')},'round':round,'original128_ms':c['median_ms']['original128'],'original256_ms':c['median_ms']['original256'],'auto_ms':c['median_ms']['auto'],'baseline_ms':base/1000,'reduction_percent':100*(1-auto/base)})
 if round==1:keys=runkeys
 else:assert keys==runkeys
 for c in r['eager']:
  for method,v in c['samples_ms'].items():assert len(v)==5 and math.isclose(statistics.median(v),c['median_ms'][method],rel_tol=1e-12)
  eager.append({**{k:c[k] for k in ('mode','dim','rows','dtype')},'round':round,'original128_ms':c['median_ms']['original128'],'auto_ms':c['median_ms']['auto'],'reduction_percent':100*(1-c['median_ms']['auto']/c['median_ms']['original128'])})
assert len(binaries)==1
def summary(items):
 groups={}
 for x in items:groups.setdefault(tuple(x[k] for k in ('mode','dim','rows','dtype')),[]).append(x)
 assert all(len(v)==3 for v in groups.values())
 return {'configurations':len(groups),'observations':len(items),'all_rounds_faster':sum(all(x['reduction_percent']>0 for x in v) for v in groups.values()),'all_rounds_at_least_5_percent':sum(all(x['reduction_percent']>=5 for x in v) for v in groups.values()),'min_reduction_percent':min(x['reduction_percent'] for x in items),'max_reduction_percent':max(x['reduction_percent'] for x in items),'nonpositive': [x for x in items if x['reduction_percent']<=0], 'under_5_percent':[x for x in items if x['reduction_percent']<5]}
report={'audit':'PASS','gpu':r['gpu'],'source_commit':validation['source_commit'],'binary_sha256':list(binaries)[0],'active_graph':summary([x for x in rows if x['activated']]),'fallback_graph':summary([x for x in rows if not x['activated']]),'eager':summary(eager),'scope':'Graph activated baseline is better same-round original128/256; fallback and eager baseline original128. Gains are time reductions, not throughput or end-to-end claims.'}
out.write_text(json.dumps(report,indent=2)+'\n')
for name,items in [('production_graph.csv',rows),('production_eager.csv',eager)]:
 with (root/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
print(json.dumps({k:({a:b for a,b in v.items() if a not in ('nonpositive','under_5_percent')} if isinstance(v,dict) else v) for k,v in report.items()},indent=2))
print('active_under_5',json.dumps(report['active_graph']['under_5_percent']))
