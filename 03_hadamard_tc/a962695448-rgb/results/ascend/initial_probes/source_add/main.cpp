#include <acl/acl.h>
#include "aclrtlaunch_ascend_add_smoke.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_ACL(call) do { const aclError code = (call); if (code != ACL_SUCCESS) \
    throw std::runtime_error(std::string(#call) + " returned " + std::to_string(code)); } while (0)

class Buffer {
public:
    void* host = nullptr;
    void* device = nullptr;
    std::size_t bytes;
    aclrtStream stream;
    Buffer(std::size_t size, aclrtStream queue) : bytes(size), stream(queue) {
        CHECK_ACL(aclrtMallocHost(&host, bytes));
        CHECK_ACL(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        std::memset(host, 0xa5, bytes);
    }
    ~Buffer() {
        if (stream) aclrtSynchronizeStream(stream);
        if (device) aclrtFree(device);
        if (host) aclrtFreeHost(host);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload() {
        CHECK_ACL(aclrtMemcpyAsync(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream));
    }
    void download() {
        CHECK_ACL(aclrtMemcpyAsync(host, bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream));
    }
};

void add_stage(aclrtStream stream) {
    constexpr std::size_t count = 256, guard = 8;
    constexpr std::size_t bytes = (count + 2 * guard) * sizeof(float);
    Buffer x(bytes, stream), y(bytes, stream), z(bytes, stream);
    auto* x_host = static_cast<float*>(x.host) + guard;
    auto* y_host = static_cast<float*>(y.host) + guard;
    std::vector<float> expected(count);
    for (std::size_t i = 0; i < count; ++i) {
        x_host[i] = (static_cast<int>(i) - 128) * 0.125f;
        y_host[i] = static_cast<int>(i % 9) * 0.25f;
        expected[i] = x_host[i] + y_host[i];
    }
    std::vector<unsigned char> x_initial(bytes), y_initial(bytes);
    std::memcpy(x_initial.data(), x.host, bytes);
    std::memcpy(y_initial.data(), y.host, bytes);
    x.upload(); y.upload(); z.upload();
    std::puts("ADD_LAUNCH mode=npu count=256 block_dim=1 payload_alignment=32 guard_bytes=32");
    CHECK_ACL(ACLRT_LAUNCH_KERNEL(ascend_add_smoke)(
        1, stream, static_cast<unsigned char*>(x.device) + guard * sizeof(float),
        static_cast<unsigned char*>(y.device) + guard * sizeof(float),
        static_cast<unsigned char*>(z.device) + guard * sizeof(float)));
    z.download(); x.download(); y.download();
    CHECK_ACL(aclrtSynchronizeStream(stream));
    if (std::memcmp(x_initial.data(), x.host, bytes) || std::memcmp(y_initial.data(), y.host, bytes))
        throw std::runtime_error("ADD input or guard modified");
    const auto* result = static_cast<const float*>(z.host) + guard;
    for (std::size_t i = 0; i < count; ++i) {
        if (std::memcmp(result + i, expected.data() + i, sizeof(float))) {
            std::printf("ADD_MISMATCH index=%zu expected=%.9g actual=%.9g\n", i, expected[i], result[i]);
            throw std::runtime_error("ADD differs from host reference");
        }
    }
    const auto* raw = static_cast<const unsigned char*>(z.host);
    for (std::size_t i = 0; i < bytes; ++i) {
        if ((i < guard * sizeof(float) || i >= (guard + count) * sizeof(float)) && raw[i] != 0xa5)
            throw std::runtime_error("ADD output guard overwritten at byte " + std::to_string(i));
    }
    std::puts("ADD_PASS elements=256 bitwise=true input_unchanged=true guards=true execution=npu");
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (!(argc == 1 || (argc == 3 && std::string(argv[1]) == "--stage" && std::string(argv[2]) == "add"))) {
        std::fprintf(stderr, "Usage: ascend_smoke [--stage add]\n");
        return 2;
    }
    bool initialized = false, device_set = false;
    aclrtStream stream = nullptr;
    try {
        CHECK_ACL(aclInit(nullptr)); initialized = true;
        CHECK_ACL(aclrtSetDevice(0)); device_set = true;
        const char* soc = aclrtGetSocName();
        std::printf("DEVICE soc=%s target=npu\n", soc ? soc : "UNKNOWN");
        CHECK_ACL(aclrtCreateStream(&stream));
        add_stage(stream);
        CHECK_ACL(aclrtDestroyStream(stream)); stream = nullptr;
        CHECK_ACL(aclrtResetDevice(0)); device_set = false;
        CHECK_ACL(aclFinalize()); initialized = false;
        std::puts("ASCEND_ADD_SMOKE_PASS short_datacopypad_and_rne=not_yet_tested");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ASCEND_ADD_SMOKE_FAIL %s\n", error.what());
        if (stream) { aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); }
        if (device_set) aclrtResetDevice(0);
        if (initialized) aclFinalize();
        return 1;
    }
}

