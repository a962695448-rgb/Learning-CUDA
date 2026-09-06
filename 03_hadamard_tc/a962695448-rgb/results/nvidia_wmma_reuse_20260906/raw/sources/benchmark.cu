#include "reuse_kernel.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do { const cudaError_t e=(call); if(e!=cudaSuccess) \
    throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(e)); } while(0)

namespace {
constexpr const char* method_names[] = {"old_wmma", "four_warp_wmma", "warp128"};
constexpr const char* source_commit = "9f5fdc363b4149d4a211701f24ab0548084ca3e5";
std::uint32_t float_bits(float x) { std::uint32_t b; std::memcpy(&b,&x,4); return b; }
template<class T> std::uint16_t storage_bits(T x) {
    static_assert(sizeof(T)==2); std::uint16_t b; std::memcpy(&b,&x,2); return b;
}
std::string escaped(const std::string& s) {
    std::string out; for(char c:s) {
        if(c=='"'||c=='\\') out+='\\';
        if(c=='\n') out+="\\n"; else if(c=='\r') out+="\\r"; else out+=c;
    } return out;
}
struct Options {
    std::size_t rows=17; int n=256, rounds=3, samples=20, iterations=100, warmup=10;
    std::string dtype="fp16", scale_kind="unit", prefix;
};
int integer(const char* value) {
    std::size_t consumed=0; const long n=std::stol(value,&consumed);
    if(consumed!=std::strlen(value)||n<1||n>1000000) throw std::invalid_argument("invalid positive integer");
    return static_cast<int>(n);
}
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(i+1>=argc) throw std::invalid_argument("option needs value: "+arg);
        const char* value=argv[++i];
        if(arg=="--rows")o.rows=integer(value); else if(arg=="--n")o.n=integer(value);
        else if(arg=="--rounds")o.rounds=integer(value); else if(arg=="--samples")o.samples=integer(value);
        else if(arg=="--iterations")o.iterations=integer(value); else if(arg=="--warmup")o.warmup=integer(value);
        else if(arg=="--dtype")o.dtype=value; else if(arg=="--scale")o.scale_kind=value;
        else if(arg=="--output-prefix")o.prefix=value; else throw std::invalid_argument("unknown option: "+arg);
    }
    if(o.n<16||o.n>256||(o.n&(o.n-1))||o.rows>16384||o.rounds!=3||o.prefix.empty()||
       (o.dtype!="fp16"&&o.dtype!="bf16")||(o.scale_kind!="unit"&&o.scale_kind!="normalized"))
        throw std::invalid_argument("requires N16..256 power2, M1..16384, 3 rounds, fp16/bf16, unit/normalized, output-prefix");
    if(o.samples>1000||o.iterations>10000||o.warmup>10000) throw std::invalid_argument("timing count exceeds experiment bound");
    for(const char* suffix:{".csv",".json"})
        if(std::filesystem::exists(o.prefix+suffix)) throw std::invalid_argument("refusing to overwrite existing results");
    return o;
}
struct Stream {
    cudaStream_t value{};
    Stream(){CHECK_CUDA(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream(){if(value)cudaStreamDestroy(value);}
};
struct Events {
    cudaEvent_t start{},stop{};
    Events(){CHECK_CUDA(cudaEventCreate(&start));CHECK_CUDA(cudaEventCreate(&stop));}
    ~Events(){if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);}
};
template<class T> class Guarded {
    std::uint8_t* base{};
    std::vector<std::uint8_t> shadow;
    std::size_t prefix, count;
public:
    Guarded(std::size_t size,int prefix_elements,cudaStream_t stream):
      shadow((size+prefix_elements+16)*sizeof(T),0xa5),prefix(prefix_elements*sizeof(T)),count(size) {
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&base),shadow.size()));
        CHECK_CUDA(cudaMemsetAsync(base,0xa5,shadow.size(),stream));
    }
    Guarded(const Guarded&)=delete; Guarded& operator=(const Guarded&)=delete;
    ~Guarded(){if(base)cudaFree(base);}
    T* data(){return reinterpret_cast<T*>(base+prefix);}
    void upload(const std::vector<T>& values,cudaStream_t stream) {
        if(values.size()!=count)throw std::logic_error("upload size mismatch");
        std::memcpy(shadow.data()+prefix,values.data(),count*sizeof(T));
        CHECK_CUDA(cudaMemcpyAsync(data(),values.data(),count*sizeof(T),cudaMemcpyHostToDevice,stream));
    }
    std::vector<T> read(cudaStream_t stream,const std::string& label,bool immutable=false) {
        std::vector<std::uint8_t> got(shadow.size());
        CHECK_CUDA(cudaMemcpyAsync(got.data(),base,got.size(),cudaMemcpyDeviceToHost,stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        for(std::size_t i=0;i<got.size();++i) {
            const bool payload=i>=prefix&&i<prefix+count*sizeof(T);
            if((immutable||!payload)&&got[i]!=shadow[i])
                throw std::runtime_error(label+": "+(payload?"input modified":"guard overwritten")+" allocation byte="+std::to_string(i));
        }
        std::vector<T> values(count);std::memcpy(values.data(),got.data()+prefix,count*sizeof(T));return values;
    }
};
template<class T> std::vector<T> make_input(std::size_t rows,int n) {
    std::vector<T> input(rows*n);
    for(std::size_t r=0;r<rows;++r)for(int j=0;j<n;++j) {
        std::uint32_t hash=std::uint32_t(r)*747796405u+std::uint32_t(j)*2891336453u+0x96269544u;
        hash^=hash>>16;hash*=2246822519u;hash^=hash>>13;
        float x=float(int(hash%1025)-512)/128.0f;
        if(r%8==1)x=0; else if(r%8==2)x=1;
        else if(r%8==3)x=(j==int(r%n)?-3.0f:0.0f);
        else if(r%8==4)x=(j&1)?-0.75f:0.75f;
        else if(r%8==5)x=(j&1)?-4.0f:4.0f-1.0f/128.0f;
        input[r*n+j]=hadamard::as_storage<T>(x);
    }return input;
}
template<class T> std::vector<T> make_general_input(std::size_t rows,int n) {
    std::vector<T> input(rows*n);
    // Fixed 24-bit uniform mantissas, with independent exponents -12..0.
    // No assumption of exact FP32 partial sums for this separate input group.
    std::uint32_t state=0x6e4d21b3u;
    for(std::size_t i=0;i<input.size();++i) {
        state^=state<<13;state^=state>>17;state^=state<<5;
        const float mantissa=float(state>>8)*(1.0f/8388608.0f)-1.0f;
        state^=state<<13;state^=state>>17;state^=state<<5;
        input[i]=hadamard::as_storage<T>(std::ldexp(mantissa,int(state%13)-12));
    }return input;
}
struct Oracle {std::vector<std::uint16_t> bits;std::vector<double> exact;};
template<class T> Oracle dense_oracle(const std::vector<T>& input,int n,float scale) {
    Oracle out;out.bits.resize(input.size());out.exact.resize(input.size());
    std::vector<double> x(input.size());
    for(std::size_t i=0;i<x.size();++i)x[i]=hadamard::as_float(input[i]);
    std::vector<double> signs(n*n);
    for(int j=0;j<n;++j)for(int k=0;k<n;++k) {
        unsigned v=j&k;bool odd=false;while(v){odd=!odd;v&=v-1;}signs[j*n+k]=odd?-1.0:1.0;
    }
    // Independent dense O(N^2) signs/dot product, not the GPU butterfly.
    // These bounded dyadic inputs make every partial FP32 sum exact. Thus the
    // required FP32 scale multiply + dtype RNE also has an exact bit oracle.
    for(std::size_t r=0;r<input.size()/n;++r)for(int j=0;j<n;++j) {
        double sum=0;for(int k=0;k<n;++k)sum+=x[r*n+k]*signs[j*n+k];
        if(double(float(sum))!=sum)throw std::logic_error("input no longer satisfies exact-dyadic oracle premise");
        const float scaled=float(sum)*scale;
        out.bits[r*n+j]=storage_bits(hadamard::as_storage<T>(scaled));out.exact[r*n+j]=sum*double(scale);
    }return out;
}
template<class T,int N> void launch(int method,const T* input,const T* matrix,T* output,
                                  std::size_t rows,float scale,cudaStream_t stream) {
    if(method==0)hadamard::tensor_core_kernel<T,N><<<dim3((rows+15)/16,N/16),128,0,stream>>>(input,matrix,output,rows,scale);
    else if(method==1)wmma_reuse::four_warp_kernel<T,N><<<dim3((rows+15)/16,(N+63)/64),128,0,stream>>>(input,matrix,output,rows,scale);
    else hadamard::warp_kernel<T,N,true,false><<<(rows+3)/4,128,0,stream>>>(input,output,nullptr,nullptr,rows,scale);
}
template<class T> void dispatch(int method,const T* input,const T* matrix,T* output,
                               const Options& o,float scale,cudaStream_t stream) {
    switch(o.n) {
#define CASE(N) case N:launch<T,N>(method,input,matrix,output,o.rows,scale,stream);break
        CASE(16);CASE(32);CASE(64);CASE(128);CASE(256);
#undef CASE
        default:throw std::logic_error("unsupported dimension");
    }
}
struct ErrorReport {double max_abs=0,max_rel=0;std::size_t output_comparisons=0;};
struct GeneralReport {
    std::size_t dense_rows=0, old_new_element_comparisons=0;
    std::array<double,3> rounded_max_abs{},unrounded_max_abs{};
    std::array<std::size_t,3> rounded_bit_mismatches{};
};
template<class T> GeneralReport validate_general(std::array<Guarded<T>*,3> outputs,
        Guarded<T>& input,Guarded<T>& matrix,const std::vector<T>& host,const Options& o,
        float scale,cudaStream_t stream) {
    GeneralReport report;report.old_new_element_comparisons=host.size();
    std::array<std::vector<T>,3> values;
    input.read(stream,"general input",true);matrix.read(stream,"general H",true);
    for(int m=0;m<3;++m) {
        values[m]=outputs[m]->read(stream,std::string("general ")+method_names[m]);
        for(T x:values[m])if(!std::isfinite(hadamard::as_float(x)))throw std::runtime_error("nonfinite general output");
    }
    if(std::memcmp(values[0].data(),values[1].data(),host.size()*sizeof(T))!=0)
        throw std::runtime_error("general input: four-warp not bitwise equal to old WMMA");
    const std::size_t samples=std::min<std::size_t>(32,o.rows);report.dense_rows=samples;
    const double tolerance=o.dtype=="fp16"?1e-2:5e-2;
    for(std::size_t s=0;s<samples;++s) {
        const std::size_t row=o.rows<=32?s:s*(o.rows-1)/31;
        for(int j=0;j<o.n;++j) {
            double sum=0;
            for(int k=0;k<o.n;++k) {
                unsigned v=j&k;bool odd=false;while(v){odd=!odd;v&=v-1;}
                sum+=(odd?-1.0:1.0)*double(hadamard::as_float(host[row*o.n+k]));
            }
            const double exact=sum*double(scale);
            const T rounded=hadamard::as_storage<T>(float(exact));
            const double expected=hadamard::as_float(rounded);
            for(int m=0;m<3;++m) {
                const T actual=values[m][row*o.n+j];
                const double value=hadamard::as_float(actual),error=std::abs(value-expected);
                report.rounded_max_abs[m]=std::max(report.rounded_max_abs[m],error);
                report.unrounded_max_abs[m]=std::max(report.unrounded_max_abs[m],std::abs(value-exact));
                report.rounded_bit_mismatches[m]+=storage_bits(actual)!=storage_bits(rounded);
                // Same strict rounded-reference threshold as pinned main.cu:310/326.
                if(!(error<tolerance))throw std::runtime_error(std::string("general ")+method_names[m]+" rounded FP64 tolerance failed row="+std::to_string(row)+" column="+std::to_string(j)+" error="+std::to_string(error));
            }
        }
    }return report;
}
template<class T> void validate(std::array<Guarded<T>*,3> outputs,Guarded<T>& input,Guarded<T>& matrix,
                               const Oracle& oracle,cudaStream_t stream,const std::string& phase,ErrorReport& errors) {
    input.read(stream,phase+" input",true);matrix.read(stream,phase+" H",true);
    std::array<std::vector<T>,3> values;
    for(int method=0;method<3;++method) {
        values[method]=outputs[method]->read(stream,phase+" "+method_names[method]);
        for(std::size_t i=0;i<values[method].size();++i) {
            const float value=hadamard::as_float(values[method][i]);
            if(!std::isfinite(value)||storage_bits(values[method][i])!=oracle.bits[i])
                throw std::runtime_error(phase+" "+method_names[method]+" dense rounded bit mismatch index="+std::to_string(i)+
                  " expected="+std::to_string(oracle.bits[i])+" actual="+std::to_string(storage_bits(values[method][i])));
            const double error=std::abs(double(value)-oracle.exact[i]);
            errors.max_abs=std::max(errors.max_abs,error);
            errors.max_rel=std::max(errors.max_rel,error/std::max(1.0,std::abs(oracle.exact[i])));
            ++errors.output_comparisons;
        }
    }
    if(std::memcmp(values[0].data(),values[1].data(),values[0].size()*sizeof(T))!=0)
        throw std::runtime_error(phase+" four-warp is not bitwise equal to old WMMA");
}
template<class T> void run(const Options& o) {
    cudaDeviceProp prop{};int device=0,runtime=0,driver=0;
    CHECK_CUDA(cudaGetDevice(&device));CHECK_CUDA(cudaGetDeviceProperties(&prop,device));
    CHECK_CUDA(cudaRuntimeGetVersion(&runtime));CHECK_CUDA(cudaDriverGetVersion(&driver));
    if(prop.major<8||prop.warpSize!=32)throw std::runtime_error("requires NVIDIA compute capability >=8 and warp32 for both dtypes");
    Stream stream;Events events;
    const float scale=o.scale_kind=="unit"?1.0f:float(1.0/std::sqrt(double(o.n)));
    auto host_input=make_input<T>(o.rows,o.n);
    std::vector<T> h(o.n*o.n);
    for(int r=0;r<o.n;++r)for(int c=0;c<o.n;++c) {
        unsigned v=r&c;bool odd=false;while(v){odd=!odd;v&=v-1;}h[r*o.n+c]=hadamard::as_storage<T>(odd?-1.0f:1.0f);
    }
    const Oracle oracle=dense_oracle(host_input,o.n,scale);ErrorReport errors;
    Guarded<T> input(host_input.size(),16,stream.value),matrix(h.size(),16,stream.value);
    Guarded<T> old_output(host_input.size(),16,stream.value),new_output(host_input.size(),16,stream.value),warp_output(host_input.size(),16,stream.value);
    std::array<Guarded<T>*,3> outputs{{&old_output,&new_output,&warp_output}};
    input.upload(host_input,stream.value);matrix.upload(h,stream.value);
    // Matrix pointers must meet WMMA alignment; ordinary I/O must also work at 2B offsets.
    for(int m=0;m<3;++m){dispatch(m,input.data(),matrix.data(),outputs[m]->data(),o,scale,stream.value);CHECK_CUDA(cudaGetLastError());}
    validate(outputs,input,matrix,oracle,stream.value,"aligned_precheck",errors);
    {
        Guarded<T> mis_input(host_input.size(),17,stream.value),a(host_input.size(),17,stream.value),b(host_input.size(),17,stream.value),c(host_input.size(),17,stream.value);
        std::array<Guarded<T>*,3> mis_outputs{{&a,&b,&c}};mis_input.upload(host_input,stream.value);
        for(int m=0;m<3;++m){dispatch(m,mis_input.data(),matrix.data(),mis_outputs[m]->data(),o,scale,stream.value);CHECK_CUDA(cudaGetLastError());}
        validate(mis_outputs,mis_input,matrix,oracle,stream.value,"offset34_precheck",errors);
    }
    // Separate non-exact data group. No requirement that warp128 match WMMA bits.
    const auto general_input=make_general_input<T>(o.rows,o.n);
    input.upload(general_input,stream.value);
    for(int m=0;m<3;++m){dispatch(m,input.data(),matrix.data(),outputs[m]->data(),o,scale,stream.value);CHECK_CUDA(cudaGetLastError());}
    const GeneralReport general=validate_general(outputs,input,matrix,general_input,o,scale,stream.value);
    input.upload(host_input,stream.value); // restore the fixed timing distribution
    std::ofstream csv(o.prefix+".csv");if(!csv)throw std::runtime_error("cannot create CSV");
    csv<<"round,position,method,sample,rows,n,dtype,scale_kind,scale_float_bits,threads,grid_x,grid_y,shared_bytes,input_offset_bytes,iterations,event_elapsed_ms,kernel_ms,timer,validation_passed\n";
    csv<<std::setprecision(17);
    for(int round=0;round<o.rounds;++round) {
        for(int position=0;position<3;++position) {
            const int m=(round+position)%3;
            for(int w=0;w<o.warmup;++w)dispatch(m,input.data(),matrix.data(),outputs[m]->data(),o,scale,stream.value);
        }
        CHECK_CUDA(cudaGetLastError());CHECK_CUDA(cudaStreamSynchronize(stream.value));
        for(int sample=0;sample<o.samples;++sample)for(int position=0;position<3;++position) {
            const int m=(round+position)%3;
            CHECK_CUDA(cudaEventRecord(events.start,stream.value));
            for(int it=0;it<o.iterations;++it)dispatch(m,input.data(),matrix.data(),outputs[m]->data(),o,scale,stream.value);
            CHECK_CUDA(cudaGetLastError());CHECK_CUDA(cudaEventRecord(events.stop,stream.value));
            CHECK_CUDA(cudaEventSynchronize(events.stop));float elapsed=0;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed,events.start,events.stop));
            if(!std::isfinite(elapsed)||elapsed<=0)throw std::runtime_error("nonfinite/nonpositive CUDA event duration");
            const double kernel_ms=double(elapsed)/o.iterations;
            const auto gx=m==2?(o.rows+3)/4:(o.rows+15)/16;
            const int gy=m==0?o.n/16:m==1?(o.n+63)/64:1;
            const int shared=m==2?0:16*o.n*int(sizeof(T))+(m==0?1:4)*256*int(sizeof(float));
            csv<<round+1<<','<<position+1<<','<<method_names[m]<<','<<sample+1<<','<<o.rows<<','<<o.n<<','<<o.dtype<<','<<o.scale_kind<<','<<float_bits(scale)
               <<",128,"<<gx<<','<<gy<<','<<shared<<",32,"<<o.iterations<<','<<double(elapsed)<<','<<kernel_ms<<",cuda_event_batched_launches,true\n";
        }
        validate(outputs,input,matrix,oracle,stream.value,"post_round_"+std::to_string(round+1),errors);
        csv.flush();if(!csv)throw std::runtime_error("CSV write failure");
    }
    std::ofstream report(o.prefix+".json");if(!report)throw std::runtime_error("cannot create validation JSON");
    report<<std::setprecision(17)<<"{\n  \"status\": \"PASS\",\n  \"source_commit\": \""<<source_commit<<"\",\n"
      <<"  \"device\": \""<<escaped(prop.name)<<"\",\n  \"compute_capability\": \""<<prop.major<<'.'<<prop.minor<<"\",\n"
      <<"  \"sm_count\": "<<prop.multiProcessorCount<<",\n  \"warp_size\": "<<prop.warpSize<<",\n  \"total_global_memory\": "<<prop.totalGlobalMem<<",\n"
      <<"  \"runtime_version\": "<<runtime<<",\n  \"driver_version\": "<<driver<<",\n  \"rows\": "<<o.rows<<",\n  \"n\": "<<o.n<<",\n"
      <<"  \"dtype\": \""<<o.dtype<<"\",\n  \"scale_kind\": \""<<o.scale_kind<<"\",\n  \"scale_float_bits\": "<<float_bits(scale)<<",\n"
      <<"  \"unique_shape_dtype_scale_cases\": 1,\n  \"guard_layouts\": [32,34],\n  \"post_round_rechecks\": 3,\n"
      <<"  \"four_warp_bitwise_equal_old_wmma\": true,\n  \"all_methods_dense_rounded_bitwise\": true,\n"
      <<"  \"oracle\": \"dense FP64 sum -> FP32 scaling -> dtype RNE; bounded dyadic inputs with exact FP32 partial sums\",\n"
      <<"  \"input_generator\": \"dyadic_v1_seed_0x96269544\",\n  \"input_and_H_unchanged\": true,\n  \"output_guards_intact\": true,\n"
      <<"  \"general_input_group\": {\n    \"generator\": \"uniform24_v1_seed_0x6e4d21b3_exponents_minus12_to_0\",\n"
      <<"    \"four_warp_bitwise_equal_old_wmma\": true,\n    \"old_new_element_comparisons\": "<<general.old_new_element_comparisons<<",\n"
      <<"    \"dense_rows\": "<<general.dense_rows<<",\n    \"strict_rounded_fp64_tolerance\": "<<(o.dtype=="fp16"?0.01:0.05)<<",\n"
      <<"    \"guard_layout_bytes\": 32,\n    \"input_and_H_unchanged\": true,\n    \"output_guards_intact\": true,\n"
      <<"    \"method_order\": [\"old_wmma\",\"four_warp_wmma\",\"warp128\"],\n"
      <<"    \"rounded_max_abs_error\": ["<<general.rounded_max_abs[0]<<','<<general.rounded_max_abs[1]<<','<<general.rounded_max_abs[2]<<"],\n"
      <<"    \"unrounded_max_abs_error\": ["<<general.unrounded_max_abs[0]<<','<<general.unrounded_max_abs[1]<<','<<general.unrounded_max_abs[2]<<"],\n"
      <<"    \"rounded_bit_mismatches\": ["<<general.rounded_bit_mismatches[0]<<','<<general.rounded_bit_mismatches[1]<<','<<general.rounded_bit_mismatches[2]<<"]\n  },\n"
      <<"  \"repeated_output_element_comparisons\": "<<errors.output_comparisons<<",\n  \"max_unrounded_fp64_abs_error\": "<<errors.max_abs<<",\n"
      <<"  \"max_unrounded_fp64_relative_to_max_1\": "<<errors.max_rel<<",\n  \"rounds\": "<<o.rounds<<",\n  \"samples_per_method_round\": "<<o.samples<<",\n"
      <<"  \"iterations_per_event\": "<<o.iterations<<",\n  \"warmup_per_method_round\": "<<o.warmup<<",\n  \"raw_event_rows\": "<<o.rounds*3*o.samples<<",\n"
      <<"  \"timing_excludes\": [\"H construction\",\"allocation\",\"H2D/D2H\",\"validation\",\"warmup\"],\n"
      <<"  \"round_process_scope\": \"three rounds in one configuration process\",\n"
      <<"  \"timer\": \"CUDA event elapsed ms / batched launch count; no CUDA Graph\"\n}\n";
    report.flush();if(!report)throw std::runtime_error("validation JSON write failure");
    std::cout<<"PASS "<<o.dtype<<" M="<<o.rows<<" N="<<o.n<<" scale="<<o.scale_kind<<" old/new bits + dense oracle + guards; "<<o.rounds*3*o.samples<<" event rows\n";
}
} // namespace
int main(int argc,char** argv) {
    Options options;
    try {options=parse(argc,argv);if(options.dtype=="fp16")run<half>(options);else run<__nv_bfloat16>(options);return 0;}
    catch(const std::exception& e) {
        std::cerr<<"FAIL "<<e.what()<<'\n';
        if(!options.prefix.empty()) {
            std::ofstream report(options.prefix+".json");
            report<<"{\"status\":\"FAIL\",\"error\":\""<<escaped(e.what())<<"\"}\n";
        }return 1;
    }
}
