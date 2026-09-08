#include <cassert>
#include <cstddef>
#include <iostream>
#include "baseline_row_policy.hpp"
#include "../../Learning-CUDA/03_hadamard_tc/a962695448-rgb/include/row_policy.hpp"

int main() {
    const char* devices[] = {"NVIDIA A800-SXM4-40GB", "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 4090 D", "NVIDIA A100-SXM4-80GB", "A100-SXM4-40GB", "Unknown GPU"};
    const char* layouts[] = {"original", "packed", "auto"};
    const std::size_t sizes[] = {1, 2, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128,
        255, 256, 257, 1024, 4095, 4096, 4097, 16383, 16384, 16385, 32769};
    const int dims[] = {0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256};
    std::size_t checked = 0;
    for (auto device : devices) for (auto layout : layouts) for (auto rows : sizes)
        for (auto dim : dims) for (auto fused : {false, true}) for (auto threads : {128, 256}) {
            auto before = baseline::choose_rows(baseline::parse_row_layout(layout), baseline::row_device(device), rows, dim, fused, threads);
            auto after = hadamard::choose_rows(hadamard::parse_row_layout(layout), hadamard::row_device(device), rows, dim, fused, threads);
            assert(before.packed == after.packed && before.threads == after.threads);
            ++checked;
        }
    auto a100 = hadamard::row_device("NVIDIA A100-SXM4-40GB");
    assert(a100 == hadamard::RowDevice::A100);
    auto below = hadamard::choose_rows(hadamard::RowLayout::Auto, a100, 255, 2, false, 128);
    auto at = hadamard::choose_rows(hadamard::RowLayout::Auto, a100, 256, 2, false, 128);
    auto rejected = hadamard::choose_rows(hadamard::RowLayout::Auto, a100, 32769, 16, true, 128);
    auto original = hadamard::choose_rows(hadamard::RowLayout::Original, a100, 4096, 2, false, 128);
    assert(!below.packed && below.threads == 128);
    assert(at.packed && at.threads == 256);
    assert(!rejected.packed && rejected.threads == 128);
    assert(!original.packed && original.threads == 128);
    std::cout << "PASS old_device_equivalence=" << checked << " a100_boundary_checks=5\n";
}
