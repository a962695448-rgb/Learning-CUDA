"""Compare the actual old/new row routing with instrumented tensor metadata."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = Path("03_hadamard_tc/a962695448-rgb")


def extract(path, name):
    source = path.read_text()
    begin = source.index("hadamard::RowChoice row_choice(")
    end = source.index("\nat::Tensor transform(", begin)
    return source[begin:end].replace("row_choice(", name + "(", 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    control = ROOT / "sources/cuda" / PROJECT
    candidate = ROOT / "candidate/cuda" / PROJECT
    assert (control / "include/row_policy.hpp").read_bytes() == (
        candidate / "include/row_policy.hpp"
    ).read_bytes()
    before = control / "src/torch_binding.cu"
    after = candidate / "src/torch_binding.cu"
    code = r"""
#include "row_policy.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

struct Reads { int sizes=0, numels=0, devices=0, properties=0; };
Reads reads;
const char* model = nullptr;
namespace at {
struct Tensor {
    std::size_t rows;
    int dim;
    long size(int axis) const {
        if (axis != -1) throw std::logic_error("Unexpected axis");
        ++reads.sizes; return dim;
    }
    std::size_t numel() const { ++reads.numels; return rows * dim; }
    int get_device() const { ++reads.devices; return 0; }
};
namespace cuda {
struct Properties { const char* name; };
const Properties* getDeviceProperties(int) {
    ++reads.properties;
    static Properties p{};
    p.name = model;
    return &p;
}
}
}
#define TORCH_CHECK(condition, message) do { if (!(condition)) \
    throw std::runtime_error(message); } while (0)
"""
    code += extract(before, "reference_choice")
    code += extract(after, "candidate_choice")
    code += r"""
struct Result {
    bool packed=false;
    int threads=0, error=0;
    std::string message;
    Reads counts;
    auto values() const { return std::tie(packed, threads, error, message); }
};
using Route = hadamard::RowChoice (*)(const at::Tensor&, const std::string&, bool, int);
Result evaluate(Route route, const at::Tensor& x, const std::string& layout,
                bool fused, int threads) {
    reads = {};
    Result r;
    try {
        const auto choice = route(x, layout, fused, threads);
        r.packed = choice.packed; r.threads = choice.threads;
    } catch (const std::invalid_argument& e) { r.error=1; r.message=e.what(); }
      catch (const std::runtime_error& e) { r.error=2; r.message=e.what(); }
    r.counts=reads;
    return r;
}
int main() {
    const char* models[] = {"NVIDIA A100-SXM4-40GB", "NVIDIA A800-SXM4-40GB",
        "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4060 Laptop GPU",
        "NVIDIA A100 80GB PCIe", "unrecognized CUDA device", ""};
    const std::size_t rows[] = {1,2,16,17,63,64,65,255,256,257,4095,4096,
        4097,16384,16385,4294967295ULL,8589934588ULL};
    const char* layouts[] = {"original", "packed", "auto", "", "Original"};
    std::size_t cases=0, rejected=0, original_reads_saved=0, device_queries_saved=0;
    for (const char* name : models)
    for (std::size_t row : rows)
    for (int dim=1; dim<=256; dim*=2)
    for (int threads : {128,256})
    for (bool fused : {false,true})
    for (const char* layout : layouts) {
        model=name;
        const at::Tensor input{row,dim};
        const auto a=evaluate(reference_choice,input,layout,fused,threads);
        const auto b=evaluate(candidate_choice,input,layout,fused,threads);
        if (a.values()!=b.values()) {
            std::cerr << "Mismatch " << name << " " << row << " " << dim
                      << " " << layout << " " << fused << " " << threads << "\n";
            return 1;
        }
        if (b.counts.sizes>a.counts.sizes || b.counts.numels>a.counts.numels ||
            b.counts.devices>a.counts.devices || b.counts.properties>a.counts.properties) {
            std::cerr << "Unexpected additional metadata reads\n"; return 1;
        }
        if (std::string(layout)=="original") {
            if (b.counts.sizes || b.counts.numels || b.counts.properties) return 1;
            original_reads_saved+=a.counts.sizes+a.counts.numels;
        }
        if (std::string(layout)=="auto" && dim>16) {
            if (a.counts.properties!=1 || b.counts.properties!=0) return 1;
            ++device_queries_saved;
        }
        ++cases; rejected+=a.error!=0;
    }
    std::cout << "{\"status\":\"PASS\",\"cases\":" << cases
              << ",\"matching_rejections\":" << rejected
              << ",\"original_metadata_reads_eliminated\":" << original_reads_saved
              << ",\"unneeded_device_queries_eliminated\":" << device_queries_saved
              << ",\"gpu_test\":false}\n";
}
"""
    source = output / "compare_policy.cpp"
    source.write_text(code)
    executable = output / "compare_policy"
    build = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-I" + str(control / "include"),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "build.log").write_text(build.stdout + build.stderr)
    build.check_returncode()
    run = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
    (output / "run.log").write_text(run.stdout + run.stderr)
    run.check_returncode()
    result = json.loads(run.stdout)
    result.update(
        control_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(after.read_bytes()).hexdigest(),
        scope="Actual extracted host routing, valid power-of-two dimensions and validated input shapes; tensor metadata is instrumented, not ATen/CUDA.",
        performance_claim="None. Real GPU/API correctness and wall-clock measurements are still required.",
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
