import argparse,json,math,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'project/scripts'))
import compare_reference as reference
import verify_block_threads as threads_check
from build_torch_extension import load_extension

class CheckedRows(threads_check.CheckedInterface):
    def hadamard(self,values,scale=1.0):
        expected=super().hadamard(values,scale)
        for threads in (128,256):
            actual=self.extension.hadamard(values,scale,threads,'auto')
            threads_check.bitwise_equal(self.torch,actual,expected,'auto transform')
            if values.shape[-1]<=16:threads_check.bitwise_equal(self.torch,self.extension.hadamard(values,scale,threads,'packed'),expected,'packed transform')
        return actual
    def hadamard_int4(self,values,scale=1.0):
        expected=super().hadamard_int4(values,scale)
        for threads in (128,256):
            actual=self.extension.hadamard_int4(values,scale,threads,'original','auto')
            threads_check.bitwise_equal(self.torch,actual,expected,'auto fused')
            if values.shape[-1]<=16:threads_check.bitwise_equal(self.torch,self.extension.hadamard_int4(values,scale,threads,'original','packed'),expected,'packed fused')
        return actual

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-repo',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();a.out.mkdir(parents=True)
    report={'status':'RUNNING','scope':'Original 1800 reference matrix, repeated auto and packed calls are not new cases; plus separately listed large-row and stream boundaries.'};code=1
    try:
        import torch
        op=load_extension(verbose=True,build_directory=str(ROOT/'build/production'));checked=CheckedRows(torch,op)
        reference.load_extension=lambda *args,**kwargs:checked
        args=argparse.Namespace(device='cuda:0',reference_repo=a.reference_repo,verbose=False,build_directory=str(ROOT/'build/row-api'),json=str(a.out/'reference.json'),benchmark=False)
        assert reference.run(args,report)==0 and report['summary']['cases']==1800
        report['large_and_stream_cases']=[]
        with torch.inference_mode():
            for n in (1,2,4,8,16):
                for dtype in (torch.float16,torch.bfloat16):
                    for rows in (15,16,17,63,64,65,255,256,257,4095,4096,4097,16384,32769):
                        for offset in (0,1):
                            stream=torch.cuda.Stream()
                            with torch.cuda.stream(stream):
                                base=torch.arange(rows*n+16,device='cuda',dtype=torch.float32).remainder(31).sub(15).div(16).to(dtype)
                                x=base[offset:offset+rows*n].reshape(1,1,rows,n);before=base.clone();scale=1/math.sqrt(n)
                                expected=op.hadamard(x,scale);expected_q=op.hadamard_int4(x,scale)
                                for layout in ('packed','auto'):
                                    for threads in (128,256):
                                        threads_check.bitwise_equal(torch,op.hadamard(x,scale,threads,layout),expected,'large stream transform')
                                        threads_check.bitwise_equal(torch,op.hadamard_int4(x,scale,threads,'original',layout),expected_q,'large stream fused')
                                threads_check.bitwise_equal(torch,base,before,'input guards')
                            stream.synchronize();report['large_and_stream_cases'].append({'n':n,'dtype':str(dtype),'rows':rows,'offset_bytes':offset*2})
        report['rejections']=[]
        sample=torch.ones((3,16),device='cuda',dtype=torch.float16)
        for method in ('hadamard','hadamard_int4'):
            for bad in ('','bad',1,None,True):
                try:getattr(op,method)(sample,row_layout=bad)
                except (TypeError,ValueError,RuntimeError):report['rejections'].append([method,repr(bad)])
                else:raise AssertionError('row layout accepted')
            for n in (32,64,128,256):
                try:getattr(op,method)(torch.ones((3,n),device='cuda',dtype=torch.float16),row_layout='packed')
                except (TypeError,ValueError,RuntimeError):report['rejections'].append([method,n])
                else:raise AssertionError('packed dimension accepted')
            for layout in ('packed','auto'):
                try:getattr(op,method)(torch._neg_view(sample),row_layout=layout)
                except RuntimeError as error:assert 'resolve_neg' in str(error);report['rejections'].append([method,'lazy_negative',layout])
                else:raise AssertionError('lazy negative accepted')
        for layout in ('packed','auto'):
            try:op.hadamard_int4(torch.ones((3,256),device='cuda',dtype=torch.float16),fused_layout='contiguous256',row_layout=layout)
            except (TypeError,ValueError,RuntimeError):report['rejections'].append(['fused_conflict',layout])
            else:raise AssertionError('conflicting layouts accepted')
        report['status']='PASS';code=0
    except Exception as e:report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());print(report['traceback'],flush=True)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'summary':report.get('summary'),'large_stream_cases':len(report.get('large_and_stream_cases',[])),'rejections':len(report.get('rejections',[]))}),flush=True);return code
if __name__=='__main__':raise SystemExit(main())
