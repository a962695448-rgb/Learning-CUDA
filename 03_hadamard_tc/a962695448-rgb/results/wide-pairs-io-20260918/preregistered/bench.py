"""One fresh process of the complete wide-transform timing matrix."""
import argparse
import functools
import gc
import hashlib
import importlib.util
import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,required=True);args=parser.parse_args();round_id=args.round;BASE=ROOT
protocol=json.loads((ROOT/'PROTOCOL.json').read_text())['cuda']
affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[0]})
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0,str(BASE/'candidate/cuda/03_hadamard_tc/a962695448-rgb/scripts'))
from verify_out_buffers import buffers_for
from verify_execution_context import compare
names={'control':'wi_control_a','twin':'wi_control_b','candidate':'wi_candidate'}
orders=[('control','candidate','twin'),('candidate','twin','control'),('twin','control','candidate')]
modules={}
for label in orders[round_id]:
    name=names[label];folder='control_a' if label=='control' else 'control_b' if label=='twin' else label
    path=BASE/'build'/folder/(name+'.so');spec=importlib.util.spec_from_file_location(name,path);op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op);modules[label]=op

def configurations():
    for case in protocol['cases']:
        yield case['kind'],case['rows'],case['dim'],case['target'],case['threads'],case['input_offset'],case['output_offset'],case['host_methods']


def host_calls_for(rows,dim):
    return max(protocol['min_host_calls'],min(protocol['host_calls'],protocol['host_element_budget']//(rows*dim)))


def call(op,x,kind,threads,buffers=None):
    name={'packed_quant':'quantize_int4_packed','original_quant':'quantize_int4',
          'packed_hadamard':'hadamard','packed_fused':'hadamard_int4',
          'original_hadamard':'hadamard','original_fused':'hadamard_int4','contiguous_fused':'hadamard_int4'}[kind]
    options={'block_threads':threads}
    if kind.endswith(('hadamard','fused')):
        options.update(scale=1.,row_layout='packed' if kind.startswith('packed_') else 'original')
    if kind=='contiguous_fused':options['fused_layout']='contiguous256'
    return functools.partial(getattr(op,name+('_out' if buffers is not None else '')),x,*(buffers or ()),**options)


def case_buffers(x,kind,offset):
    if not offset:return buffers_for(torch,x,kind)[0],[]
    specs=[(tuple(x.shape),x.dtype)] if kind=='hadamard' else [((*x.shape[:-1],x.shape[-1]//2),torch.uint8),(tuple(x.shape[:-1]),torch.float32)]
    buffers=[];guards=[]
    for shape,dtype in specs:
        base=torch.full((math.prod(shape)+17,),37,dtype=dtype,device=x.device)
        buffers.append(base[9:-8].reshape(shape));guards.append(base)
    return tuple(buffers),guards


def check_guards(guards):
    for guard in guards:assert torch.all(guard[:9]==37) and torch.all(guard[-8:]==37)


def graph_for(fn):
    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(30):fn()
    side.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=side):
        for _ in range(protocol['graph_nodes']):fn()
    torch.cuda.synchronize()
    for _ in range(10):graph.replay()
    torch.cuda.synchronize()
    return graph


def device_time(graph):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(protocol['graph_replays']):graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)*1000/(protocol['graph_nodes']*protocol['graph_replays'])


def host_sample(fn,calls):
    torch.cuda.synchronize()
    usage=resource.getrusage(resource.RUSAGE_SELF)
    collections=[g['collections'] for g in gc.get_stats()]
    cpu0=time.process_time_ns();thread0=time.thread_time_ns();start=time.perf_counter_ns()
    for _ in range(calls):fn()
    torch.cuda.synchronize()
    end=time.perf_counter_ns();thread1=time.thread_time_ns();cpu1=time.process_time_ns()
    after=resource.getrusage(resource.RUSAGE_SELF)
    return {'start_ns':start,'end_ns':end,'wall_us':(end-start)/1000/calls,
      'process_cpu_us':(cpu1-cpu0)/1000/calls,'thread_cpu_us':(thread1-thread0)/1000/calls,
      'voluntary_switches':after.ru_nvcsw-usage.ru_nvcsw,'involuntary_switches':after.ru_nivcsw-usage.ru_nivcsw,
      'gc_collections':[g['collections']-n for g,n in zip(gc.get_stats(),collections)]}


def spread(values):
    q1,_,q3=statistics.quantiles(values,n=4,method='inclusive')
    return 100*(q3-q1)/statistics.median(values)


report={'status':'RUNNING','round':round_id,'load_order':orders[round_id],
  'cpu_affinity':sorted(os.sched_getaffinity(0)),'records':[],'host_records':[],'groups':[],'calibration':[],
  'protocol':protocol,'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
  'gpu':torch.cuda.get_device_name(),'binaries':{v:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for v,m in modules.items()}}
path=ROOT/'results-remote'/f'round-{round_id}.json'


def save():
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(path)


save()
rng=torch.Generator().manual_seed(protocol['seed']+round_id)
for dtype in (torch.float16,torch.bfloat16):
    for kind,rows,dim,target,thread_choices,input_offset,output_offset,host_methods in configurations():
        calls=host_calls_for(rows,dim)
        cpu=torch.randn((rows,dim),generator=rng).to(dtype)
        input_pool=None
        if input_offset:
            input_pool=torch.full((cpu.numel()+2,),37,dtype=dtype,device='cuda')
            x=input_pool[1:-1].reshape(cpu.shape);x.copy_(cpu)
            assert x.data_ptr()%4==2
        else:x=cpu.cuda()
        input_copy=x.cpu();input_version=x._version
        buffer_kind='hadamard' if kind.endswith('hadamard') else 'quantize_int4'
        for threads in thread_choices:
            shared,guards=case_buffers(x,buffer_kind,output_offset)
            fns={v:call(modules[v],x,kind,threads,shared) for v in ('control','candidate')}
            expected=call(modules['control'],x,kind,threads)();expected=expected if isinstance(expected,tuple) else (expected,)
            expected=tuple(t.cpu() for t in expected)
            graphs={v:graph_for(fn) for v,fn in fns.items()}
            for v in ('control','candidate'):
                graphs[v].replay();torch.cuda.synchronize();compare(torch,shared,expected)
            times={v:[] for v in fns};device_order=[]
            for group in range(protocol['groups']):
                order=('control','candidate') if (group+round_id)%2==0 else ('candidate','control')
                device_order.append(list(order))
                for v in order:times[v].append(device_time(graphs[v]))
            ratio=statistics.median(times['control'])/statistics.median(times['candidate'])
            case={'round':round_id,'dtype':str(dtype),'kind':kind,'rows':rows,'dim':dim,'threads':threads,'target':target,'host_calls':calls,'input_offset':input_offset,'output_offset':output_offset}
            report['records'].append(dict(case,device_us=times,order=device_order,speedup=ratio,gate_passed=ratio>=1/1.05))
            for v in ('control','candidate'):
                graphs[v].replay();torch.cuda.synchronize();compare(torch,shared,expected)
            del graphs
            for method in host_methods:
                funcs={v:call(modules[v],x,kind,threads,shared if method=='out' else None) for v in ('control','candidate')}
                for fn in funcs.values():
                    value=fn();torch.cuda.synchronize();compare(torch,shared if method=='out' else value,expected);del value
                    for _ in range(100):fn()
                samples={v:[] for v in funcs};timeline=[]
                for group in range(protocol['groups']):
                    order=('control','candidate') if (group+round_id)%2==0 else ('candidate','control')
                    for v in order:
                        measurement=host_sample(funcs[v],calls);samples[v].append(measurement['wall_us'])
                        timeline.append(dict(measurement,group=group,variant=v,order=list(order)))
                for fn in funcs.values():
                    value=fn();torch.cuda.synchronize();compare(torch,shared if method=='out' else value,expected);del value
                ratio=statistics.median(samples['control'])/statistics.median(samples['candidate'])
                paired=statistics.median(a/b for a,b in zip(samples['control'],samples['candidate']))
                variability={v:spread(values) for v,values in samples.items()}
                paired_variability=spread([a/b for a,b in zip(samples['control'],samples['candidate'])])
                passed=ratio>=1/1.05 and paired>=1/1.05 and paired_variability<=protocol['max_paired_iqr_percent']
                report['host_records'].append(dict(case,method=method,host_us=samples,timeline=timeline,
                  speedup=ratio,paired_speedup=paired,iqr_percent=variability,paired_iqr_percent=paired_variability,absolute_spread_flag=max(variability.values())>protocol['max_host_iqr_percent'],shared_output_objects=method=='out',gate_passed=passed))
            check_guards(guards)
        compare(torch,x,(input_copy,));assert x._version==input_version
        if input_pool is not None:assert input_pool[0].item()==37 and input_pool[-1].item()==37
        save();print(round_id,str(dtype),kind,rows,dim,'complete',flush=True)
for dtype,kind,rows,dim,threads,method in protocol['same_source_aa_cases']:
    calls=host_calls_for(rows,dim)
    x=torch.randn((rows,dim),generator=rng).to(getattr(torch,dtype.removeprefix('torch.'))).cuda()
    shared=buffers_for(torch,x,'hadamard' if kind.endswith('hadamard') else 'quantize_int4')[0]
    funcs={v:call(modules['control' if v=='a' else 'twin'],x,kind,threads,shared if method=='out' else None) for v in ('a','b')}
    expected=call(modules['control'],x,kind,threads)();expected=expected if isinstance(expected,tuple) else (expected,);expected=tuple(t.cpu() for t in expected)
    for fn in funcs.values():
        value=fn();torch.cuda.synchronize();compare(torch,shared if method=='out' else value,expected);del value
        for _ in range(1000):fn()
    samples={v:[] for v in funcs};timeline=[]
    for group in range(protocol['groups']):
        order=('a','b') if (group+round_id)%2==0 else ('b','a')
        for v in order:
            measurement=host_sample(funcs[v],calls);samples[v].append(measurement['wall_us'])
            timeline.append(dict(measurement,group=group,variant=v,order=list(order)))
    ratio=statistics.median(samples['a'])/statistics.median(samples['b'])
    paired=statistics.median(a/b for a,b in zip(samples['a'],samples['b']))
    variability={v:spread(values) for v,values in samples.items()}
    paired_variability=spread([a/b for a,b in zip(samples['a'],samples['b'])])
    passed=1/1.05<=ratio<=1.05 and 1/1.05<=paired<=1.05 and paired_variability<=protocol['max_paired_iqr_percent']
    report['calibration'].append({'round':round_id,'dtype':dtype,'kind':kind,'rows':rows,'dim':dim,'threads':threads,'method':method,'host_calls':calls,
      'host_us':samples,'timeline':timeline,'same_source_ratio':ratio,'paired_ratio':paired,'iqr_percent':variability,'paired_iqr_percent':paired_variability,'absolute_spread_flag':max(variability.values())>protocol['max_host_iqr_percent'],'gate_passed':passed})
for dtype in ('torch.float16','torch.bfloat16'):
    for threads in protocol['threads']:
        for kind in protocol['target_kinds']:
            records=[r for r in report['records'] if r['dtype']==dtype and r['threads']==threads and r['kind']==kind and r['target']]
            ratio=math.exp(statistics.mean(math.log(r['speedup']) for r in records))
            report['groups'].append({'round':round_id,'dtype':dtype,'threads':threads,'kind':kind,'target_geomean_speedup':ratio,
              'gate_passed':ratio>=protocol['minimum_each_kind_dtype_thread_round_target_geomean']})
report['status']='ACCEPT' if all(r['gate_passed'] for key in ('records','host_records','groups','calibration') for r in report[key]) else 'REJECT'
save();print('ROUND_END',round_id,report['status'],flush=True)

