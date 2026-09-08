from pathlib import Path
import csv,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parent;P=ROOT/'project';OUT=ROOT/'runs/validation';OUT.mkdir(parents=True,exist_ok=False)
report={'status':'RUNNING','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'steps':[],'checks':{}}
def save():(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
def verify():
    m=json.loads((ROOT/'production_manifest.json').read_text())
    for n,item in m['files'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==item['sha256'],n
    return m
def run(name,argv,expected=0):
    entry={'name':name,'argv':list(map(str,argv))};report['steps'].append(entry);save()
    with (OUT/(name+'.log')).open('xb') as log:
        p=subprocess.Popen(entry['argv'],cwd=P,stdout=log,stderr=subprocess.STDOUT);entry['pid']=p.pid;save();entry['exit_code']=p.wait()
    save();assert entry['exit_code']==expected,name
    return (OUT/(name+'.log')).read_text(errors='replace')
try:
    m=verify();report['source_commit']=m['source_commit'];save()
    arch='89' if str(ROOT).startswith('/data') else '80'
    run('build',['make','-j1','ARCH='+arch,'CUDA_HOME=/usr/local/cuda','all','cpu-test'])
    binary=P/'build/hadamard'
    run('legacy_cli',[sys.executable,P/'scripts/run_validation.py','--label','rows_default'])
    text=(P/'results/validation_rows_default.log').read_text();assert 'SELF_TEST PASS cases=1876' in text and text.count('EXIT_CODE 2;')==15
    modes=[('original256',['--block-threads','256']),('contiguous256',['--fused-layout','contiguous256']),('packed128',['--row-layout','packed']),('packed256',['--row-layout','packed','--block-threads','256']),('auto128',['--row-layout','auto']),('auto256',['--row-layout','auto','--block-threads','256'])]
    for name,args in modes:assert 'SELF_TEST PASS cases=1876' in run(name,[binary,'--self-test',*args])
    report['checks']['cli_modes']=7;report['checks']['cases_per_cli_mode']=1876
    invalid=[['--row-layout','bad'],['--row-layout',''],['--row-layout'],['--benchmark','--row-layout','packed','--dim','64'],['--row-layout','auto','--fused-layout','contiguous256'],['--row-layout','packed','--fused-layout','contiguous256']]
    for i,args in enumerate(invalid):run('row_reject_'+str(i),[binary,*args],2)
    report['checks']['row_cli_rejections']=len(invalid)
    run('row_api',[sys.executable,ROOT/'validate_row_api.py','--reference-repo',os.environ['REFERENCE_REPO'],'--out',OUT/'row-api'])
    api=json.loads((OUT/'row-api/report.json').read_text());assert api['status']=='PASS' and api['summary']['cases']==1800
    report['checks'].update(reference_cases=1800,large_stream_cases=len(api['large_and_stream_cases']),row_api_rejections=len(api['rejections']))
    run('metadata',[sys.executable,P/'scripts/verify_tensor_metadata.py','--build-directory',ROOT/'build/production','--json',OUT/'metadata.json'])
    meta=json.loads((OUT/'metadata.json').read_text());assert meta['status']=='PASS' and len(meta['cases'])==28
    report['checks']['metadata_cases']=28
    for layout,columns in [('original',21),('packed',24),('auto',24)]:
        csvpath=OUT/(layout+'.csv')
        run('csv_'+layout,[binary,'--benchmark','--seq','4096','--batch','1','--heads','1','--dim','16','--row-layout',layout,'--warmup','0','--repetitions','1','--csv',csvpath])
        with csvpath.open() as f:r=csv.DictReader(f);rows=list(r);assert len(r.fieldnames)==columns and len(rows)==7
        if layout!='original':
            for row in rows:
                assert row['requested_row_layout']==layout
                if row['method']=='split_int4':assert row['quantize_block_threads']=='128'
        for row in rows:assert abs(float(row['mean_ms'])-float(row['mean_us'])/1000)<1e-9
    legacy=OUT/'original.csv';before=hashlib.sha256(legacy.read_bytes()).hexdigest()
    run('reject_mixed_csv',[binary,'--benchmark','--dim','16','--row-layout','auto','--repetitions','1','--warmup','0','--csv',legacy],1)
    assert hashlib.sha256(legacy.read_bytes()).hexdigest()==before
    report['checks']['csv_contract']='PASS';verify();report['status']='PASS'
except Exception as e:report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());print(report['traceback'],flush=True)
report['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save();print(json.dumps(report),flush=True)
