from pathlib import Path
import csv,hashlib,json,math,statistics,sys,zipfile
root=Path(__file__).resolve().parent;device=sys.argv[1];archive=root/(device+'-final-metrics.zip');target=root/'metrics'/device;target.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archive) as z:
 m=json.loads(z.read('manifest.json'))
 for name,item in m['files'].items():
  data=z.read(name);assert hashlib.sha256(data).hexdigest()==item['sha256'] and len(data)==item['bytes'];dest=(target/name).resolve();assert dest.is_relative_to(target.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
policy=json.loads((target/'final_policy.json').read_text());validation=target/'production/runs/validation/report.json';assert json.loads(validation.read_text())['status']=='PASS'
rows=[];eager=[];keys=None;binary=set()
pairs=[('remove_keyword_auto','auto_kwrow','auto_pos_full'),('auto_vs_original_positional','original_pos_full','auto_pos_full'),('auto_positional_vs_legacy','original_pos3','auto_pos_full')]
for round in (1,2,3):
 r=json.loads((target/f'runs/graph{round}.json').read_text());assert r['status']=='PASS' and r['source_commit']=='24dfef776ad73a4128cb6138674c5886c21e49c0';binary.add(r['binary_sha256'])
 assert r['validation_sha256']==hashlib.sha256(validation.read_bytes()).hexdigest();rk=set()
 for c in r['benchmarks']:
  k=tuple(c[x] for x in ('dim','rows','mode','dtype','scale'));assert k not in rk;rk.add(k)
  assert c['independent_output_buffers'] and c['cross_method_output_pointers_disjoint'] and c['outputs_bitwise_equal_eager_before_and_after']
  assert c['captured_calls_per_graph']==64 and c['replays_per_group']==20 and c['graph_warmup_replays']==256
  for name,intervals in c['raw_event_intervals_ms'].items():
   assert len(intervals)==5 and all(x>0 for x in intervals);values=[x*1000/1280 for x in intervals]
   assert all(math.isclose(a,b,rel_tol=1e-12) for a,b in zip(values,c['samples_us'][name]));assert math.isclose(statistics.median(values),c['median_us'][name],rel_tol=1e-12)
  active=any(x['dim']==c['dim'] and x['mode']==c['mode'] and c['rows']>=x['min_rows'] for x in policy['rules']);assert active==c['activated']
  baseline=min(c['median_us']['original128'],c['median_us']['original256']) if active else c['median_us']['original128'];auto=c['median_us']['auto']
  rows.append({**{x:c[x] for x in ('dim','rows','mode','dtype','scale','activated')},'round':round,'original128_ms':c['median_us']['original128']/1000,'original256_ms':c['median_us']['original256']/1000,'auto_ms':auto/1000,'baseline_ms':baseline/1000,'reduction_percent':100*(1-auto/baseline)})
 if keys is None:keys=rk
 else:assert keys==rk
 e=json.loads((target/f'eager/round{round}.json').read_text());assert e['status']=='PASS';binary.add(e['binary_sha256'])
 for c in e['cases']:
  assert c['outputs_bitwise_equal'] and c['input_unchanged']
  for name,ns in c['raw_batch_ns'].items():
   assert len(ns)==9;values=[x/2000/1e6 for x in ns];assert all(math.isclose(a,b,rel_tol=1e-12) for a,b in zip(values,c['samples_ms'][name]));assert math.isclose(statistics.median(values),c['median_ms'][name],rel_tol=1e-12)
  for label,b,a in pairs:eager.append({**{x:c[x] for x in ('dim','rows','mode','dtype')},'round':round,'comparison':label,'baseline_ms':c['median_ms'][b],'candidate_ms':c['median_ms'][a],'reduction_percent':100*(1-c['median_ms'][a]/c['median_ms'][b])})
assert len(binary)==1
def summary(items):
 groups={}
 for x in items:groups.setdefault(tuple(x[k] for k in ('dim','rows','mode','dtype')),[]).append(x)
 assert all(len(v)==3 for v in groups.values())
 return {'configs':len(groups),'observations':len(items),'all_rounds_faster':sum(all(x['reduction_percent']>0 for x in v) for v in groups.values()),'all_rounds_ge5':sum(all(x['reduction_percent']>=5 for x in v) for v in groups.values()),'min':min(x['reduction_percent'] for x in items),'max':max(x['reduction_percent'] for x in items),'under5':[x for x in items if x['reduction_percent']<5]}
report={'audit':'PASS','gpu':policy['gpu'],'source_commit':'24dfef776ad73a4128cb6138674c5886c21e49c0','binary_sha256':list(binary)[0],'graph_active':summary([x for x in rows if x['activated']]),'graph_fallback':summary([x for x in rows if not x['activated']]),'eager':{label:summary([x for x in eager if x['comparison']==label]) for label,_,_ in pairs}}
(root/(device+'_final_analysis.json')).write_text(json.dumps(report,indent=2)+'\n')
for name,items in [('graph',rows),('eager',eager)]:
 with (root/(device+'_final_'+name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
def compact(x):return {k:compact(v) if isinstance(v,dict) else v for k,v in x.items() if k!='under5'}
print(json.dumps(compact(report),indent=2));print('active_under5',json.dumps(report['graph_active']['under5']))
