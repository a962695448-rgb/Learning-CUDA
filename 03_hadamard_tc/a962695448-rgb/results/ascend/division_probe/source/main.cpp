#include <acl/acl.h>
#include "aclrtlaunch_ascend_div_vector_probe.h"
#if PROBE_SCALAR_DIV
#include "aclrtlaunch_ascend_div_scalar_probe.h"
#endif

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#pragma STDC FENV_ACCESS ON
#define CHECK_ACL(call) do { const aclError error = (call); if (error != ACL_SUCCESS) \
    throw std::runtime_error(std::string(#call) + " returned " + std::to_string(error)); } while (0)

namespace {
constexpr std::size_t kCount = 256, kGuard = 8;
uint32_t bits(float value) { uint32_t result; std::memcpy(&result, &value, sizeof(result)); return result; }
float from_bits(uint32_t value) { float result; std::memcpy(&result, &value, sizeof(result)); return result; }

// volatile阻止常量折叠；host目标禁用fast-math，存回FP32再读出。
__attribute__((noinline)) float host_div(float numerator, float denominator) {
    volatile float lhs = numerator, rhs = denominator;
    volatile float quotient = lhs / rhs;
    return quotient;
}

struct Case { float lhs, rhs, expected; const char* group; };
std::vector<Case> make_cases() {
    const std::array<uint32_t, 32> magnitudes{{
        0x00000000, 0x33800000, 0x38800000, 0x3b800000, 0x3d000000, 0x3d800000, 0x3e000000, 0x3e800000,
        0x3f000000, 0x3f800000, 0x3f802000, 0x3f810000, 0x3fc00000, 0x40000000, 0x40400000, 0x40800000,
        0x40e00000, 0x41000000, 0x41800000, 0x42000000, 0x42800000, 0x42ff0000, 0x437f0000, 0x43800000,
        0x44000000, 0x44800000, 0x45000000, 0x45800000, 0x477fe000, 0x0d800000, 0x71800000, 0x7f7f0000}};
    std::vector<Case> cases;
    auto add = [&](float lhs, float rhs, const char* group) {
        const float expected = host_div(lhs, rhs);
        if (!std::isfinite(lhs) || !std::isfinite(rhs) || rhs == 0 || !std::isfinite(expected) ||
            (lhs != 0 && std::fpclassify(lhs) != FP_NORMAL) || std::fpclassify(rhs) != FP_NORMAL ||
            (expected != 0 && std::fpclassify(expected) != FP_NORMAL))
            throw std::runtime_error("fixture outside finite normal-or-zero scope");
        cases.push_back({lhs, rhs, expected, group});
    };
    for (uint32_t value : magnitudes) add(from_bits(value), 7.0f, "maxabs_over_7");
    for (uint32_t value : magnitudes) add(from_bits(value ^ 0x80000000u), 7.0f, "negative_over_7_control");
    const std::array<int, 16> indices{{1, 2, 4, 7, 8, 9, 10, 11, 12, 14, 16, 21, 28, 29, 30, 31}};
    const std::array<float, 8> factors{{0.0f, -0.0f, 1.0f, -1.0f, .5f, -.5f, .25f, -.25f}};
    for (int index : indices) {
        const float magnitude = from_bits(magnitudes[index]);
        const float scale = host_div(magnitude, 7.0f);
        for (float factor : factors) add(magnitude * factor, scale, "x_over_cpu_fp32_scale");
    }
    const std::array<uint32_t, 8> numerators{{0x00000000, 0x80000000, 0x3f800000, 0xbf800000,
                                          0x40400000, 0xc0400000, 0x477fe000, 0xc77fe000}};
    const std::array<uint32_t, 8> denominators{{0x3f800000, 0xbf800000, 0x40400000, 0xc0400000,
                                            0x40e00000, 0xc0e00000, 0x3f810000, 0x3eaaa000}};
    for (uint32_t lhs : numerators)
        for (uint32_t rhs : denominators) add(from_bits(lhs), from_bits(rhs), "general_sign_control");
    if (cases.size() != kCount) throw std::runtime_error("fixture count differs from kernel count");
    return cases;
}

class Buffer {
public:
    void* host = nullptr;
    void* device = nullptr;
    aclrtStream stream;
    static constexpr std::size_t bytes = (kCount + 2 * kGuard) * sizeof(float);
    explicit Buffer(aclrtStream queue) : stream(queue) {
        CHECK_ACL(aclrtMallocHost(&host, bytes));
        const aclError code = aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (code != ACL_SUCCESS) { aclrtFreeHost(host); host = nullptr; CHECK_ACL(code); }
        std::memset(host, 0xa5, bytes);
    }
    ~Buffer() {
        if (stream) aclrtSynchronizeStream(stream);
        if (device) aclrtFree(device);
        if (host) aclrtFreeHost(host);
    }
    Buffer(const Buffer&) = delete;
    float* payload() { return static_cast<float*>(host) + kGuard; }
    unsigned char* device_payload() { return static_cast<unsigned char*>(device) + kGuard * sizeof(float); }
    void upload() { CHECK_ACL(aclrtMemcpyAsync(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream)); }
    void download() { CHECK_ACL(aclrtMemcpyAsync(host, bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream)); }
};

struct Result { std::size_t mismatches = 0, signed_zero_only = 0; std::vector<float> actual; };
Result measure_path(aclrtStream stream, const std::vector<Case>& cases, bool scalar) {
    const char* path = scalar ? "aicore_cpp_div" : "vector_div";
    Buffer lhs(stream), rhs(stream), output(stream);
    for (std::size_t i = 0; i < kCount; ++i) {
        lhs.payload()[i] = cases[i].lhs;
        rhs.payload()[i] = cases[i].rhs;
    }
    std::vector<unsigned char> lhs_before(Buffer::bytes), rhs_before(Buffer::bytes);
    std::memcpy(lhs_before.data(), lhs.host, Buffer::bytes);
    std::memcpy(rhs_before.data(), rhs.host, Buffer::bytes);
    lhs.upload(); rhs.upload(); output.upload();
    std::printf("DIV_LAUNCH path=%s execution=npu count=256 block_dim=1 guard_bytes=32\n", path);
#if PROBE_SCALAR_DIV
    if (scalar)
        CHECK_ACL(ACLRT_LAUNCH_KERNEL(ascend_div_scalar_probe)(1, stream, lhs.device_payload(), rhs.device_payload(), output.device_payload()));
    else
#else
    if (scalar) throw std::runtime_error("scalar requested from vector-only executable");
#endif
        CHECK_ACL(ACLRT_LAUNCH_KERNEL(ascend_div_vector_probe)(1, stream, lhs.device_payload(), rhs.device_payload(), output.device_payload()));
    output.download(); lhs.download(); rhs.download();
    CHECK_ACL(aclrtSynchronizeStream(stream));
    if (std::memcmp(lhs_before.data(), lhs.host, Buffer::bytes) || std::memcmp(rhs_before.data(), rhs.host, Buffer::bytes))
        throw std::runtime_error(std::string(path) + " input/guard modified");
    const auto* raw = static_cast<const unsigned char*>(output.host);
    for (std::size_t i = 0; i < Buffer::bytes; ++i)
        if ((i < kGuard * sizeof(float) || i >= (kGuard + kCount) * sizeof(float)) && raw[i] != 0xa5)
            throw std::runtime_error(std::string(path) + " output guard overwritten at byte " + std::to_string(i));
    Result result;
    result.actual.assign(output.payload(), output.payload() + kCount);
    std::array<std::size_t, 4> group_mismatches{{0, 0, 0, 0}};
    std::size_t printed = 0;
    for (std::size_t i = 0; i < kCount; ++i) {
        const bool mismatch = bits(result.actual[i]) != bits(cases[i].expected);
        if (mismatch) {
            ++result.mismatches;
            ++group_mismatches[i < 32 ? 0 : (i < 64 ? 1 : (i < 192 ? 2 : 3))];
            if (result.actual[i] == 0 && cases[i].expected == 0) ++result.signed_zero_only;
        }
        if (i < 4 || (mismatch && printed < 16)) {
            std::printf("DIV_%s path=%s index=%zu group=%s lhs=%.9g rhs=%.9g expected=%.9g actual=%.9g lhs_bits=%08x rhs_bits=%08x expected_bits=%08x actual_bits=%08x\n",
                mismatch ? "MISMATCH" : "SAMPLE", path, i, cases[i].group, cases[i].lhs, cases[i].rhs,
                cases[i].expected, result.actual[i], bits(cases[i].lhs), bits(cases[i].rhs), bits(cases[i].expected), bits(result.actual[i]));
            if (mismatch) ++printed;
        }
    }
    std::printf("DIV_RESULT path=%s elements=256 mismatches=%zu signed_zero_only=%zu maxabs_over_7=%zu negative_control=%zu x_over_scale=%zu general_control=%zu input_unchanged=true guards=true status=%s\n",
        path, result.mismatches, result.signed_zero_only, group_mismatches[0], group_mismatches[1], group_mismatches[2], group_mismatches[3],
        result.mismatches ? "PRECISION_MISMATCH" : "BITWISE_PASS");
    return result;
}
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string mode = PROBE_SCALAR_DIV ? "both" : "vector";
    if (argc == 3 && std::string(argv[1]) == "--mode") mode = argv[2];
    else if (argc != 1) { std::fprintf(stderr, "Usage: ascend_div_probe [--mode vector|scalar|both]\n"); return 2; }
    if ((mode != "vector" && mode != "scalar" && mode != "both") || (!PROBE_SCALAR_DIV && mode != "vector")) {
        std::fprintf(stderr, "Unsupported mode; scalar/both requires a separate ENABLE_SCALAR_DIV=ON build\n"); return 2;
    }
    bool initialized = false, device_set = false;
    aclrtStream stream = nullptr;
    try {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "host IEEE FP32 required");
        if (std::fesetround(FE_TONEAREST) != 0 || std::fegetround() != FE_TONEAREST ||
            bits(host_div(1.0f, 7.0f)) != 0x3e124925u || bits(host_div(1.0f, 2.0f)) != 0x3f000000u ||
            bits(host_div(from_bits(0x80000000u), 7.0f)) != 0x80000000u)
            throw std::runtime_error("host FP32 RNE calibration failed");
        const auto cases = make_cases();
        std::printf("DIV_PROBE_BEGIN mode=%s scalar_compiled=%d host_rne_checks=3 cases=256\n", mode.c_str(), PROBE_SCALAR_DIV);
        CHECK_ACL(aclInit(nullptr)); initialized = true;
        CHECK_ACL(aclrtSetDevice(0)); device_set = true;
        const char* soc = aclrtGetSocName();
        std::printf("DEVICE soc=%s execution=npu\n", soc ? soc : "UNKNOWN");
        CHECK_ACL(aclrtCreateStream(&stream));
        Result vector, scalar;
        if (mode != "scalar") vector = measure_path(stream, cases, false);
        if (mode != "vector") scalar = measure_path(stream, cases, true);
        if (mode == "both") {
            std::size_t differences = 0;
            for (std::size_t i = 0; i < kCount; ++i) differences += bits(vector.actual[i]) != bits(scalar.actual[i]);
            std::printf("DIV_PATH_COMPARISON elements=256 vector_vs_scalar_bit_mismatches=%zu\n", differences);
        }
        CHECK_ACL(aclrtDestroyStream(stream)); stream = nullptr;
        CHECK_ACL(aclrtResetDevice(0)); device_set = false;
        CHECK_ACL(aclFinalize()); initialized = false;
        const bool pass = vector.mismatches == 0 && scalar.mismatches == 0;
        const char* vector_status = mode == "scalar" ? "NOT_TESTED" : (vector.mismatches ? "PRECISION_MISMATCH" : "BITWISE_PASS");
        const char* scalar_status = mode == "vector" ? "NOT_TESTED" : (scalar.mismatches ? "PRECISION_MISMATCH" : "BITWISE_PASS");
        std::printf("ASCEND_DIV_PROBE_RESULT mode=%s status=%s vector_status=%s scalar_status=%s vector_mismatches=%zu scalar_mismatches=%zu\n",
            mode.c_str(), pass ? "BITWISE_PASS" : "PRECISION_MISMATCH", vector_status, scalar_status, vector.mismatches, scalar.mismatches);
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ASCEND_DIV_PROBE_ERROR %s\n", error.what());
        if (stream) { aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); }
        if (device_set) aclrtResetDevice(0);
        if (initialized) aclFinalize();
        return 1;
    }
}
