#include <acl/acl.h>
#include "aclrtlaunch_ascend_pad_fp16.h"
#include "aclrtlaunch_ascend_pad_bf16.h"
#include "aclrtlaunch_ascend_rne_fp16.h"
#include "aclrtlaunch_ascend_rne_bf16.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_ACL(call) do { const aclError code=(call); if(code!=ACL_SUCCESS) \
    throw std::runtime_error(std::string(#call)+" returned "+std::to_string(code)); } while(0)

uint32_t bits(float x) { uint32_t b; std::memcpy(&b,&x,4); return b; }
float value(uint32_t b) { float x; std::memcpy(&x,&b,4); return x; }
uint32_t rshift_rne(uint32_t x, unsigned s) {
    const auto q=x>>s, rem=x&((1u<<s)-1), mid=1u<<(s-1);
    return q+(rem>mid || (rem==mid && (q&1)));
}
uint16_t half_bits(float x) {
    const auto b=bits(x), f=b&0x7fffff;
    const auto sign=static_cast<uint16_t>((b>>16)&0x8000);
    const int e=static_cast<int>((b>>23)&255)-127;
    if(e>15) return sign|0x7c00;
    if(e<-25) return sign;
    if(e<-14) return sign|rshift_rne(f|0x800000,-e-1);
    return sign|(((e+15)<<10)+rshift_rne(f,13));
}
uint16_t bf_bits(float x) {
    const uint32_t b=bits(x), hi=b>>16, lo=b&0xffff;
    return static_cast<uint16_t>(hi+(lo>0x8000 || (lo==0x8000 && (hi&1))));
}
float half_value(uint16_t b) {
    const int e=(b>>10)&31, f=b&1023;
    const float x=e ? std::ldexp(static_cast<float>(1024+f),e-25) : std::ldexp(static_cast<float>(f),-24);
    return b&0x8000 ? -x : x;
}

class Guarded {
    void* host_=nullptr; void* device_=nullptr;
    std::size_t prefix_, payload_, total_;
    aclrtStream stream_;
    std::vector<unsigned char> original_;
public:
    Guarded(std::size_t count,std::size_t width,aclrtStream stream)
        :prefix_(17*width),payload_(count*width),total_((payload_+2*prefix_+63)/64*64),
         stream_(stream),original_(total_,0xa5) {
        CHECK_ACL(aclrtMallocHost(&host_,total_));
        CHECK_ACL(aclrtMalloc(&device_,total_,ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ~Guarded() {
        if(stream_) aclrtSynchronizeStream(stream_);
        if(device_) aclrtFree(device_);
        if(host_) aclrtFreeHost(host_);
    }
    Guarded(const Guarded&)=delete; Guarded& operator=(const Guarded&)=delete;
    unsigned char* device() { return static_cast<unsigned char*>(device_)+prefix_; }
    const unsigned char* payload() const { return static_cast<const unsigned char*>(host_)+prefix_; }
    void upload(const void* src=nullptr) {
        if(src) std::memcpy(original_.data()+prefix_,src,payload_);
        std::memcpy(host_,original_.data(),total_);
        CHECK_ACL(aclrtMemcpyAsync(device_,total_,host_,total_,ACL_MEMCPY_HOST_TO_DEVICE,stream_));
    }
    void check(const char* label,bool unchanged=false) {
        CHECK_ACL(aclrtMemcpyAsync(host_,total_,device_,total_,ACL_MEMCPY_DEVICE_TO_HOST,stream_));
        CHECK_ACL(aclrtSynchronizeStream(stream_));
        const auto* b=static_cast<const unsigned char*>(host_);
        for(std::size_t i=0;i<total_;++i) {
            if((unchanged || i<prefix_ || i>=prefix_+payload_) && b[i]!=original_[i])
                throw std::runtime_error(std::string(label)+" guard/input mismatch at byte "+std::to_string(i));
        }
    }
};

void pad_case(aclrtStream stream,bool bf,uint32_t n) {
    std::vector<uint16_t> input(n);
    for(uint32_t i=0;i<n;++i) input[i]=static_cast<uint16_t>(i*251u+0x3517u);
    Guarded x(n,2,stream), y(n,2,stream);
    x.upload(input.data()); y.upload();
    std::printf("PAD_LAUNCH dtype=%s n=%u gm_offset_bytes=34\n",bf?"bf16":"fp16",n);
    if(bf) CHECK_ACL(ACLRT_LAUNCH_KERNEL(ascend_pad_bf16)(1,stream,x.device(),y.device(),n));
    else CHECK_ACL(ACLRT_LAUNCH_KERNEL(ascend_pad_fp16)(1,stream,x.device(),y.device(),n));
    x.check("pad_input",true); y.check("pad_output");
    if(std::memcmp(input.data(),y.payload(),n*2)) throw std::runtime_error("DataCopyPad payload differs");
    std::printf("PAD_PASS dtype=%s n=%u bits_exact=true input_unchanged=true guards=true\n",bf?"bf16":"fp16",n);
}

void rne_case(aclrtStream stream,bool bf,uint32_t n) {
    const float even=bf?1.00390625f:1.00048828125f;
    const float odd=bf?1.01171875f:1.00146484375f;
    const std::vector<float> edges{even,odd,-even,-odd,std::nextafter(even,0.0f),
        std::nextafter(even,2.0f),std::nextafter(-even,0.0f),std::nextafter(-even,-2.0f),
        0.0f,-0.0f,1.0f,-1.0f,0.00006103515625f,0.000000059604644775390625f};
    std::vector<float> input(n),expected_float(n);
    std::vector<uint16_t> expected(n);
    uint32_t seed=0x7351a723;
    for(uint32_t i=0;i<n;++i) {
        seed=seed*1664525u+1013904223u;
        input[i]=i<edges.size()?edges[(i+n)%edges.size()]:(static_cast<float>(seed>>8)/16777216.0f-0.5f)*32.0f;
        expected[i]=bf?bf_bits(input[i]):half_bits(input[i]);
        expected_float[i]=bf?value(static_cast<uint32_t>(expected[i])<<16):half_value(expected[i]);
    }
    Guarded x(n,4,stream), y(n,2,stream), back(n,4,stream);
    x.upload(input.data()); y.upload(); back.upload();
    std::printf("RNE_LAUNCH dtype=%s n=%u input_offset=68 typed_output_offset=34\n",bf?"bf16":"fp16",n);
    if(bf) CHECK_ACL(ACLRT_LAUNCH_KERNEL(ascend_rne_bf16)(1,stream,x.device(),y.device(),back.device(),n));
    else CHECK_ACL(ACLRT_LAUNCH_KERNEL(ascend_rne_fp16)(1,stream,x.device(),y.device(),back.device(),n));
    x.check("rne_input",true); y.check("rne_stored"); back.check("rne_readback");
    for(uint32_t i=0;i<n;++i) {
        uint16_t actual; uint32_t actual_float;
        std::memcpy(&actual,y.payload()+i*2,2); std::memcpy(&actual_float,back.payload()+i*4,4);
        if(actual!=expected[i] || actual_float!=bits(expected_float[i])) {
            std::printf("RNE_MISMATCH dtype=%s n=%u index=%u input=%08x expected=%04x actual=%04x expected_back=%08x actual_back=%08x\n",
                bf?"bf16":"fp16",n,i,bits(input[i]),unsigned(expected[i]),unsigned(actual),bits(expected_float[i]),actual_float);
            throw std::runtime_error("Cast RNE differs from independent integer oracle");
        }
    }
    std::printf("RNE_PASS dtype=%s n=%u bits_exact=true readback_exact=true input_unchanged=true guards=true\n",bf?"bf16":"fp16",n);
}

int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    std::string stage="all",dtype="both"; uint32_t only_n=0;
    try {
        for(int i=1;i<argc;i+=2) {
            if(i+1==argc) throw std::runtime_error("missing option value");
            const std::string key=argv[i],val=argv[i+1];
            if(key=="--stage") stage=val;
            else if(key=="--dtype") dtype=val;
            else if(key=="--n") {
                std::size_t used=0; const auto parsed=std::stoul(val,&used);
                if(used!=val.size() || parsed<1 || parsed>257) throw std::runtime_error("n must be 1..257");
                only_n=static_cast<uint32_t>(parsed);
            } else throw std::runtime_error("unknown option");
        }
        if(stage!="pad" && stage!="rne" && stage!="all") throw std::runtime_error("stage must be pad/rne/all");
        if(dtype!="fp16" && dtype!="bf16" && dtype!="both") throw std::runtime_error("dtype must be fp16/bf16/both");
    } catch(const std::exception& e) { std::fprintf(stderr,"INVALID_ARGUMENT %s\n",e.what()); return 2; }
    aclrtStream stream=nullptr; bool initialized=false,device_set=false;
    try {
        CHECK_ACL(aclInit(nullptr)); initialized=true;
        CHECK_ACL(aclrtSetDevice(0)); device_set=true;
        CHECK_ACL(aclrtCreateStream(&stream));
        const char* soc=aclrtGetSocName(); std::printf("DEVICE soc=%s execution=npu\n",soc?soc:"UNKNOWN");
        const std::vector<uint32_t> lengths=only_n?std::vector<uint32_t>{only_n}:
            std::vector<uint32_t>{1,2,3,7,8,15,16,17,31,32,63,64,127,128,255,256,257};
        unsigned passed=0;
        for(bool bf:{false,true}) {
            if((bf && dtype=="fp16") || (!bf && dtype=="bf16")) continue;
            for(auto n:lengths) {
                if(stage=="pad" || stage=="all") {pad_case(stream,bf,n);++passed;}
                if(stage=="rne" || stage=="all") {rne_case(stream,bf,n);++passed;}
            }
        }
        CHECK_ACL(aclrtDestroyStream(stream)); stream=nullptr;
        CHECK_ACL(aclrtResetDevice(0)); device_set=false;
        CHECK_ACL(aclFinalize()); initialized=false;
        std::printf("ASCEND_STAGE2_PASS cases=%u stage=%s dtype=%s execution=npu\n",passed,stage.c_str(),dtype.c_str());
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"ASCEND_STAGE2_FAIL %s\n",e.what());
        if(stream) {aclrtSynchronizeStream(stream);aclrtDestroyStream(stream);}
        if(device_set) aclrtResetDevice(0);
        if(initialized) aclFinalize();
        return 1;
    }
}

