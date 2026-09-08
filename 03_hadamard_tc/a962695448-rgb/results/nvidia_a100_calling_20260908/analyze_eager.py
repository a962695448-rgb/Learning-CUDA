from pathlib import Path
import csv,hashlib,json,math,statistics,sys,zipfile
root=Path(sys.argv[1]);p=root/'eager-results.zip';out=root/'eager-retrieved';out.mkdir(exist_ok=True)
with zipfile.ZipFile(p) as z:
 m=json.loads(z.read('manifest.json'))
 for name,item in m['files'].items():
  data=z.read(name);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
  target=(out/name).resolve();assert target.is_relative_to(out.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
protocol=json.loads((out/'eager_protocol.json').read_text());rows=[];raw=[];binaries=set();case_sets=[]
pairs=[('auto_vs_original_positional','original_pos_full','auto_pos_full'),('auto_vs_original_keyword','original_kwrow','auto_kwrow'),('remove_keyword_auto','auto_kwrow','auto_pos_full'),('original_keyword_vs_legacy','original_pos3','original_kwrow'),('auto_positional_vs_legacy','original_pos3','auto_pos_full'),('packed_vs_original_positional','original_pos_full','packed_pos_full')]
for round in (1,2,3):
 r=json.loads((out/f'eager/round{round}.json').read_text());assert r['status']=='PASS' and r['round']==round
 binaries.add(r['binary_sha256']);keys=set()
 for c in r['cases']:
  key=tuple(c[k] for k in ('dim','rows','mode','dtype'));assert key not in keys;keys.add(key)
  assert c['outputs_bitwise_equal'] and c['input_unchanged']
  for name,ns in c['raw_batch_ns'].items():
   vals=[n/protocol['timing']['calls_per_group']/1e6 for n in ns];assert len(vals)==protocol['timing']['groups']
   assert all(math.isclose(a,b,rel_tol=1e-12) for a,b in zip(vals,c['samples_ms'][name]))
   assert math.isclose(statistics.median(vals),c['median_ms'][name],rel_tol=1e-12)
   raw.append({**{k:c[k] for k in ('dim','rows','mode','dtype')},'round':round,'method':name,'median_ms':c['median_ms'][name]})
  for label,base,method in pairs:
   rows.append({**{k:c[k] for k in ('dim','rows','mode','dtype')},'round':round,'comparison':label,'baseline_ms':c['median_ms'][base],'candidate_ms':c['median_ms'][method],'reduction_percent':100*(1-c['median_ms'][method]/c['median_ms'][base])})
 case_sets.append(keys)
assert len(binaries)==1 and case_sets[0]==case_sets[1]==case_sets[2] and len(case_sets[0])==12
summary={}
for label,base,method in pairs:
 relevant=[x for x in rows if x['comparison']==label];bycase={}
 for x in relevant:bycase.setdefault(tuple(x[k] for k in ('dim','rows','mode','dtype')),[]).append(x['reduction_percent'])
 summary[label]={'min':min(x['reduction_percent'] for x in relevant),'max':max(x['reduction_percent'] for x in relevant),'all_rounds_faster':sum(min(v)>0 for v in bycase.values()),'all_rounds_at_least_5_percent':sum(min(v)>=5 for v in bycase.values())}
report={'status':'PASS','cases':12,'rounds':3,'binary_sha256':list(binaries)[0],'comparisons':summary,'scope':'Controlled eager diagnostics on A100; time reductions use same-round group medians. Does not replace original A800/4090 observations.'}
(root/'eager_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
for name,items in [('eager_differences.csv',rows),('eager_medians.csv',raw)]:
 with (root/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
print(json.dumps(report,indent=2))
