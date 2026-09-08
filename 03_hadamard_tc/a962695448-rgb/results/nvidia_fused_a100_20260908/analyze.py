"""Independently replay numeric certificates and derive all A100 timing comparisons."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
root=args.root.resolve();args.output.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reg=json.loads((root/'runs/regression/regression_report.json').read_text())
assert reg['status']=='PASS' and reg['exit_code']==0
files=json.loads((root/'run_manifest.json').read_text())['files']
for name,item in files.items():
    p=root/name;assert p.stat().st_size==item['size'] and sha(p)==item['sha256'],name
spec=importlib.util.spec_from_file_location('a100_numeric_certificate',root/'revised_v2/numeric_certificate.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
expected={(dtype,m,scale) for dtype in ('fp16','bf16') for m in [2,3,16,18,63,65,255,256,258,4095,4097,16383,16385] for scale in (1.,.0625)}
expected_conditions={(pat,seed,offset) for pat,seeds in {'normal':[2026,95811],'uniform':[2026,95811],'outlier':[2026,95811],'zeros':[2026]}.items() for seed in seeds for offset in (0,2)}
hard_counts=('certified_elements','gpu_vs_exact_direct_storage_bit_differences','exact_direct_vs_via_fp32_numeric_differences')
hard_maxima=('prestorage_error','storage_rounding_error','stored_error_vs_exact')
def key(c):return c['dtype'],c['rows'],c['scale']
cache={};results={k:[] for k in expected};raw=[];inputs={};max_abs={'fp16':0.,'bf16':0.};buffer_ids=set()
for index in (1,2,3):
    path=root/f'runs/holdout/run{index}.json';r=json.loads(path.read_text());inputs[str(path.relative_to(root))]=sha(path)
    assert r['status']=='PASS' and r['exit_code']==0 and r['run_index']==index and r['actual_gpu_execution']
    assert r['regression_report_sha256']==sha(root/'runs/regression/regression_report.json')
    assert r['source_manifest_sha256']==sha(root/'source_manifest.json')
    assert r['environment']['sm']==[8,0] and 'A100' in r['environment']['gpu']
    assert r['environment']['extension_sha256']==reg['binaries']['production_extension']['sha256']
    sample_path=path.with_name(r['sample_buffers']['file']);assert sample_path.parent==path.parent and sha(sample_path)==r['sample_buffers']['sha256']
    buffers=json.loads(sample_path.read_text());arrays={}
    for digest,item in buffers.items():
        array=np.array(item['values'],dtype='<u2');assert list(array.shape)==item['shape'] and hashlib.sha256(array.tobytes()).hexdigest()==digest
        arrays[digest]=array;buffer_ids.add(digest)
    assert len(r['correctness'])==52 and {key(c) for c in r['correctness']}==expected
    assert len(r['benchmarks'])==52 and {key(c) for c in r['benchmarks']}==expected
    for c in r['correctness']:
        assert len(c['checks'])==14 and {(v['pattern'],v['seed'],v['pointer_mod16']) for v in c['checks']}==expected_conditions
        for check in c['checks']:
            assert check['pass'] and check['official_reference']['pass'] and check['cpu_quantization_exact'] and check['input_guards_unchanged'] and check['original_candidate_fused_split_exact']
            metric=check['official_reference'];max_abs[c['dtype']]=max(max_abs[c['dtype']],metric['max_abs_error'])
            assert metric['max_abs_error'] < (.01 if c['dtype']=='fp16' else .05)
            ih,oh=check['sample_input_bits_sha256'],check['sample_gpu_bits_sha256']
            signature=(c['dtype'],c['scale'],ih,oh,tuple(check['sample_row_ids']))
            if signature not in cache:
                cache[signature]=module.certify_samples(arrays[ih],arrays[oh],c['dtype'],c['scale'],row_ids=check['sample_row_ids'])
            actual=cache[signature];recorded=check['numeric_certificate']
            assert actual['status']==recorded['status']=='PASS'
            for field in ('input_u16_le_sha256','supplied_output_u16_le_sha256','cpu_pre_f32_le_sha256'):
                assert actual[field]==recorded[field]
            for field in hard_counts:assert actual['summary']['counts'][field]==recorded['summary']['counts'][field]
            for field in hard_maxima:assert actual['summary']['maxima'][field]['fraction']==recorded['summary']['maxima'][field]['fraction']
    for c in r['benchmarks']:
        assert c['independent_output_buffers'] and c['cross_method_output_pointers_disjoint'] and c['outputs_bitwise_equal_eager_before_and_after']
        assert c['captured_calls_per_graph']==64 and c['replays_per_group']==20
        for method in ('original','contiguous256'):
            vals=c['samples_us'][method];intervals=c['raw_event_intervals_ms'][method]
            assert len(vals)==len(intervals)==5
            assert statistics.median(vals)==c['median_us'][method]
            for group,(us,ms) in enumerate(zip(vals,intervals)):
                assert math.isfinite(ms) and ms>0 and math.isclose(us,ms*1000/1280,rel_tol=1e-12)
                raw.append({'run':index,'dtype':c['dtype'],'rows':c['rows'],'scale':c['scale'],'method':method,'group':group,'event_ms':ms,'per_call_us':us,'per_call_ms':us/1000})
        reduction=(1-c['median_us']['contiguous256']/c['median_us']['original'])*100
        results[key(c)].append(reduction)
    print('VERIFIED',index,flush=True)
rows=[{'dtype':k[0],'rows':k[1],'scale':k[2],'run1_reduction_pct':v[0],'run2_reduction_pct':v[1],'run3_reduction_pct':v[2],'minimum_reduction_pct':min(v),'median_reduction_pct':statistics.median(v),'all_three_at_least_5pct':all(x>=5 for x in v),'any_over_3pct_regression':any(x < -3 for x in v)} for k,v in sorted(results.items())]
for name,data in [('comparison.csv',rows),('samples.csv',raw)]:
    with (args.output/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary={'status':'PASS','hardware':'A100-SXM4-40GB sm80','source_commit':'35f79b9cc595b0459a6172e0e2a63dcb9f4854af','distinct_configurations':52,'conditions_per_process':728,'processes':3,'raw_timing_samples':len(raw),'unique_buffers':len(buffer_ids),'replayed_cpu_certificates':len(cache),'official_max_abs_error':max_abs,'all_process_reduction_min_pct':min(x for v in results.values() for x in v),'all_process_reduction_max_pct':max(x for v in results.values() for x in v),'all_three_positive':sum(all(x>0 for x in v) for v in results.values()),'all_three_at_least_5pct':sum(all(x>=5 for x in v) for v in results.values()),'any_over_3pct_regression':[k for k,v in results.items() if any(x < -3 for x in v)],'input_hashes':inputs,'regression_report_sha256':sha(root/'runs/regression/regression_report.json'),'scope':'Same-process original-fused versus contiguous256 CUDA Graph timing; no cross-device or end-to-end denominator.'}
(args.output/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2))
