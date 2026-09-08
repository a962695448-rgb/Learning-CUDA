#include "packed_rows.cuh"
#include "reference.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#define CUDA_OK(call) do { const auto e=(call); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)

template<class T,int N> void check(int rows,int offset,int threads) {
    const int count=rows*N, bytes=rows*((N+1)/2), guard=16;
    std::vector<T> input(count+guard,hadamard::as_storage<T>(123.f));
    std::vector<float> values(count);
    for(int i=0;i<count;++i) { values[i]=float((i%31)-15)*.125f; input[offset+i]=hadamard::as_storage<T>(values[i]); }
    hadamard::fwht(values.data(),rows,N,.25f);
    for(auto& v:values) v=hadamard::as_float(hadamard::as_storage<T>(v));
    const auto quant=hadamard::quantize_int4(values,N);
    T *in=nullptr,*out=nullptr;std::uint8_t* packed=nullptr;float* scales=nullptr;
    CUDA_OK(cudaMalloc(&in,input.size()*sizeof(T)));CUDA_OK(cudaMalloc(&out,input.size()*sizeof(T)));
    CUDA_OK(cudaMalloc(&packed,bytes+guard));CUDA_OK(cudaMalloc(&scales,(rows+guard)*sizeof(float)));
    CUDA_OK(cudaMemcpy(in,input.data(),input.size()*sizeof(T),cudaMemcpyHostToDevice));
    const std::vector<T> output_guard(input.size(),hadamard::as_storage<T>(123.f));
    const std::vector<std::uint8_t> byte_guard(bytes+guard,205);
    const std::vector<float> scale_guard(rows+guard,-888.f);
    CUDA_OK(cudaMemcpy(out,output_guard.data(),output_guard.size()*sizeof(T),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(packed,byte_guard.data(),byte_guard.size(),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(scales,scale_guard.data(),scale_guard.size()*sizeof(float),cudaMemcpyHostToDevice));
    const int blocks=(rows-1)/(threads/N)+1;
    hadamard::packed_rows_kernel<T,N,true,false><<<blocks,threads>>>(in+offset,out+offset,nullptr,nullptr,rows,.25f);
    CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
    std::vector<T> actual(output_guard.size());CUDA_OK(cudaMemcpy(actual.data(),out,actual.size()*sizeof(T),cudaMemcpyDeviceToHost));
    for(int i=0;i<int(actual.size());++i) {
        T expected=(i>=offset && i<offset+count)?hadamard::as_storage<T>(values[i-offset]):output_guard[i];
        if(std::memcmp(&actual[i],&expected,sizeof(T))) throw std::runtime_error("transform or output guard mismatch");
    }
    for(int mode=0;mode<2;++mode) {
        if(mode==0) hadamard::packed_rows_kernel<T,N,true,true><<<blocks,threads>>>(in+offset,nullptr,packed+3,scales+3,rows,.25f);
        else hadamard::packed_rows_kernel<T,N,false,true><<<blocks,threads>>>(out+offset,nullptr,packed+3,scales+3,rows,1.f);
        CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
        std::vector<std::uint8_t> actual_bytes(byte_guard.size());std::vector<float> actual_scales(scale_guard.size());
        CUDA_OK(cudaMemcpy(actual_bytes.data(),packed,actual_bytes.size(),cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(actual_scales.data(),scales,actual_scales.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(int i=0;i<int(actual_bytes.size());++i) if(actual_bytes[i]!=(i>=3 && i<3+bytes?quant.packed[i-3]:byte_guard[i])) throw std::runtime_error("INT4 or packed guard mismatch");
        for(int i=0;i<int(actual_scales.size());++i) {
            float expected=i>=3 && i<3+rows?quant.scales[i-3]:scale_guard[i];
            if(std::memcmp(&actual_scales[i],&expected,sizeof(float))) throw std::runtime_error("scale or guard mismatch");
        }
    }
    std::vector<T> after(input.size());CUDA_OK(cudaMemcpy(after.data(),in,after.size()*sizeof(T),cudaMemcpyDeviceToHost));
    if(std::memcmp(after.data(),input.data(),input.size()*sizeof(T))) throw std::runtime_error("input modified");
    CUDA_OK(cudaFree(in));CUDA_OK(cudaFree(out));CUDA_OK(cudaFree(packed));CUDA_OK(cudaFree(scales));
}

int main() {
    try {
        int cases=0;
        for(int rows:{1,3,17,129}) for(int offset:{0,1}) for(int threads:{128,256}) {
#define CHECK_N(N) check<__half,N>(rows,offset,threads);check<__nv_bfloat16,N>(rows,offset,threads);cases+=2
            CHECK_N(1);CHECK_N(2);CHECK_N(4);CHECK_N(8);CHECK_N(16);
#undef CHECK_N
        }
        std::printf("MEMORY_FIXTURES_PASS cases=%d\n",cases);return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what());return 1; }
}
