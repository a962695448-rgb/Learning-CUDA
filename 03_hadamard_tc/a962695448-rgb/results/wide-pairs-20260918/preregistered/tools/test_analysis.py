"""Exercise complete-matrix auditing and the additional evidence gates."""
import copy
import json
import unittest
from pathlib import Path

from analyze import audit, configurations, host_calls_for


def timeline(times, round_id, calls):
    rows=[];cursor=0;labels=tuple(times)
    for group in range(len(times[labels[0]])):
        order=labels if (group+round_id)%2==0 else labels[::-1]
        for label in order:
            duration=round(times[label][group]*1000*calls)
            rows.append({'group':group,'variant':label,'order':list(order),'start_ns':cursor,'end_ns':cursor+duration,'wall_us':times[label][group]})
            cursor+=duration+1000
    return rows


def fixture():
    p=json.loads((Path(__file__).resolve().parents[1]/'PROTOCOL.json').read_text())['cuda']
    report={'status':'ACCEPT','protocol':p,'records':[],'host_records':[],'groups':[],'calibration':[]}
    for round_id in range(3):
        for dtype in ('torch.float16','torch.bfloat16'):
            for kind,rows,dim,target,thread_choices in configurations(p):
                calls=host_calls_for(p,rows,dim)
                for threads in thread_choices:
                    case=dict(round=round_id,dtype=dtype,kind=kind,rows=rows,dim=dim,threads=threads,target=target,host_calls=calls)
                    order=[list(('control','candidate') if (group+round_id)%2==0 else ('candidate','control')) for group in range(p['groups'])]
                    report['records'].append(dict(case,device_us={'control':[1.2]*p['groups'],'candidate':[1.]*p['groups']},order=order,speedup=1.2,gate_passed=True))
                    for method in ('allocating','out'):
                        times={'control':[10.]*p['groups'],'candidate':[10.]*p['groups']}
                        report['host_records'].append(dict(case,method=method,host_us=times,timeline=timeline(times,round_id,calls),speedup=1.,paired_speedup=1.,iqr_percent={'control':0.,'candidate':0.},paired_iqr_percent=0.,absolute_spread_flag=False,shared_output_objects=method=='out',gate_passed=True))
            for threads in (128,256):
                for kind in p['target_kinds']:
                    report['groups'].append(dict(round=round_id,dtype=dtype,threads=threads,kind=kind,target_geomean_speedup=1.2,gate_passed=True))
        for dtype,kind,rows,dim,threads,method in p['same_source_aa_cases']:
            calls=host_calls_for(p,rows,dim)
            times={'a':[10.]*p['groups'],'b':[10.]*p['groups']}
            report['calibration'].append(dict(round=round_id,dtype=dtype,kind=kind,rows=rows,dim=dim,threads=threads,method=method,host_calls=calls,host_us=times,timeline=timeline(times,round_id,calls),same_source_ratio=1.,paired_ratio=1.,iqr_percent={'a':0.,'b':0.},paired_iqr_percent=0.,absolute_spread_flag=False,gate_passed=True))
    return report,p


class AuditTests(unittest.TestCase):
    def test_full_original_matrix(self):
        data,p=fixture();result=audit(data,p)
        self.assertEqual((result['decision'],result['device_records'],result['host_records'],len(result['groups']),result['aa_cases']),('ACCEPT',678,1356,24,18))

    def test_missing_case_and_bad_timeline_fail_closed(self):
        data,p=fixture();bad=copy.deepcopy(data);bad['records'].pop()
        with self.assertRaises(AssertionError):audit(bad,p)
        data['host_records'][0]['timeline'][0]['variant']='candidate'
        with self.assertRaises(AssertionError):audit(data,p)

    def test_matched_host_regression_is_rejected(self):
        data,p=fixture();r=data['host_records'][0]
        r['host_us']['candidate']=[11.]*p['groups'];r['timeline']=timeline(r['host_us'],0,r['host_calls'])
        r.update(speedup=10/11,paired_speedup=10/11,gate_passed=False);data['status']='REJECT'
        result=audit(data,p);self.assertEqual(result['decision'],'REJECT');self.assertEqual(len(result['failed_gates']),1)

    def test_common_mode_drift_is_recorded_without_relative_instability(self):
        data,p=fixture();r=data['calibration'][0]
        r['host_us']={'a':[8.]*9+[12.]*9,'b':[8.]*9+[12.]*9};r['timeline']=timeline(r['host_us'],0,r['host_calls'])
        r['iqr_percent']={'a':40.,'b':40.};r['absolute_spread_flag']=True
        self.assertEqual(audit(data,p)['decision'],'ACCEPT')


    def test_relative_instability_still_fails(self):
        data,p=fixture();r=data['calibration'][0]
        r['host_us']={'a':[10.]*18,'b':[8.]*9+[12.]*9};r['timeline']=timeline(r['host_us'],0,r['host_calls'])
        r.update(same_source_ratio=1.,paired_ratio=(1.25+10/12)/2,iqr_percent={'a':0.,'b':40.},paired_iqr_percent=40.,absolute_spread_flag=True,gate_passed=False);data['status']='REJECT'
        self.assertEqual(audit(data,p)['decision'],'REJECT')

    def test_fixed_size_scaled_calls_and_wrong_count_fail_closed(self):
        data,p=fixture()
        for rows,dim,expected in ((1,32,5000),(16387,32,5000),(65536,256,256),(262147,256,128)):
            self.assertEqual(host_calls_for(p,rows,dim),expected)
        record=data['host_records'][0]
        record['host_calls'] *= 2
        record['timeline']=timeline(record['host_us'],0,record['host_calls'])
        with self.assertRaises(AssertionError):audit(data,p)


if __name__=='__main__':unittest.main()
