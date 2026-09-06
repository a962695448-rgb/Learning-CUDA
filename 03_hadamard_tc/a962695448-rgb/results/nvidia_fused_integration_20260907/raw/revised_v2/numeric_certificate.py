#!/usr/bin/env python3
"""CPU-only exact-rational certificate for supplied N256 FP16/BF16 samples.

Main API: certify_samples(input_bits, gpu_output_bits, dtype, scale=1.0,
                         *, dense_fp64=None, row_ids=None,
                         include_element_diagnostics=False).
Returns a JSON-serializable PASS/FAIL report. No GPU libraries or execution.
"""
from fractions import Fraction
from functools import lru_cache
import hashlib
import math
import numpy as np


N = 256
U32 = Fraction(1, 1 << 24)
ETA32 = Fraction(1, 1 << 149)
GAMMA9 = Fraction(9, (1 << 24) - 9)
FORMATS = {"fp16": (5, 10, 15), "bf16": (8, 7, 127), "fp32": (8, 23, 127)}
ASSUMPTIONS = [
    "IEEE binary32 add/subtract and one scale multiply use round-to-nearest, ties-to-even.",
    "Eight global butterfly stages execute in bit order 0 through 7; no reassociation or fused alternative arithmetic.",
    "Finite inputs, intermediates and outputs; no overflow in the certified arithmetic or storage conversion.",
    "Gradual underflow, with no flush-to-zero or denormals-are-zero; storage conversion is IEEE nearest-even.",
    "Scale is exactly 1 or 1/16. Input storage-to-binary32 conversion is exact.",
    "GPU output bits are supplied by the caller; this module neither runs a GPU nor authenticates their origin.",
]


class CertificateError(Exception):
    def __init__(self, code, **details):
        self.failure = {"code": code, **details}


def _fail(code, **details):
    raise CertificateError(code, **details)


def _pow2(exponent):
    return Fraction(1 << exponent, 1) if exponent >= 0 else Fraction(1, 1 << (-exponent))


def _fraction_record(value):
    """The float is an upward display bound, never used in a hard comparison."""
    value = Fraction(value)
    nearest = float(value)
    upper = math.nextafter(nearest, math.inf) if value else 0.0
    return {"fraction": f"{value.numerator}/{value.denominator}", "upper_float": upper}


def _bits_fraction(bits, dtype):
    exponent_bits, fraction_bits, bias = FORMATS[dtype]
    sign = -1 if bits >> (exponent_bits + fraction_bits) else 1
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits) - 1)
    tail = bits & ((1 << fraction_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        _fail("NONFINITE_IEEE_BITS", dtype=dtype, bits=hex(bits))
    mantissa = tail if exponent == 0 else (1 << fraction_bits) + tail
    power = (1 - bias if exponent == 0 else exponent - bias) - fraction_bits
    return sign * mantissa * _pow2(power)


def _floor_log2(value):
    numerator, denominator = value.numerator, value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    too_high = numerator < (denominator << exponent) if exponent >= 0 else (numerator << (-exponent)) < denominator
    return exponent - 1 if too_high else exponent


def round_fraction_to_bits(value, dtype, *, negative_zero=False):
    """Integer nearest-even rounding, including subnormals and signed zero.

    Returns the encoded infinity on overflow; certificate callers reject it.
    fp32 is available for rounding diagnostics; input/output API uses fp16/bf16.
    """
    if dtype not in FORMATS:
        raise ValueError("dtype must be fp16, bf16 or fp32")
    value = Fraction(value)
    exponent_bits, fraction_bits, bias = FORMATS[dtype]
    sign_bit = 1 << (exponent_bits + fraction_bits)
    sign = sign_bit if value < 0 or (value == 0 and negative_zero) else 0
    if value == 0:
        return sign
    magnitude = abs(value)
    exponent = max(_floor_log2(magnitude), 1 - bias)
    step = exponent - fraction_bits
    numerator, denominator = magnitude.numerator, magnitude.denominator
    if step >= 0:
        denominator <<= step
    else:
        numerator <<= -step
    integer, remainder = divmod(numerator, denominator)
    if remainder * 2 > denominator or (remainder * 2 == denominator and integer % 2):
        integer += 1
    if integer < (1 << fraction_bits):
        return sign | integer
    if integer == (1 << (fraction_bits + 1)):
        integer >>= 1
        exponent += 1
    field = exponent + bias
    if field >= (1 << exponent_bits) - 1:
        return sign | (((1 << exponent_bits) - 1) << fraction_bits)
    return sign | (field << fraction_bits) | (integer - (1 << fraction_bits))


def _u16_matrix(value, label):
    array = np.asarray(value)
    if array.ndim == 1 and array.shape == (N,):
        array = array.reshape(1, N)
    if array.ndim != 2 or array.shape[1] != N or array.shape[0] < 1:
        _fail("INVALID_SAMPLE_SHAPE", argument=label, shape=list(array.shape))
    if array.dtype.kind not in "ui" or np.any(array < 0) or np.any(array > 65535):
        _fail("INVALID_U16_BITS", argument=label)
    return np.array(array, dtype=np.uint16, order="C", copy=True)


def _decode(bits, dtype):
    if dtype == "fp16":
        return bits.view(np.float16).astype(np.float32)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _hex(bits, width=4):
    return f"{int(bits):0{width}x}"


def _check_finite_bits(bits, dtype, label, row_ids):
    exponent_bits, fraction_bits, _ = FORMATS[dtype]
    mask = ((1 << exponent_bits) - 1) << fraction_bits
    bad = (bits.astype(np.uint32) & mask) == mask
    if np.any(bad):
        row, column = map(int, np.argwhere(bad)[0])
        _fail("NONFINITE_SAMPLE", argument=label, sample_row_index=row, row=row_ids[row], column=column,
              bits=_hex(bits[row, column]))


def _check_cpu_environment():
    if np.dtype(np.float32).itemsize != 4 or np.dtype(np.float64).itemsize != 8:
        _fail("UNSUPPORTED_NUMPY_FLOAT_FORMAT")
    def bits(value):
        return int(np.asarray(value, dtype=np.float32).view(np.uint32))
    with np.errstate(all="ignore"):
        one = np.float32(1)
        half_ulp = np.float32(2.0 ** -24)
        odd = np.array(0x3F800001, dtype=np.uint32).view(np.float32)[()]
        minimum = np.array(1, dtype=np.uint32).view(np.float32)[()]
        checks = {
            "nearest_even_lower": bits(np.add(one, half_ulp, dtype=np.float32)) == 0x3F800000,
            "nearest_even_upper": bits(np.add(odd, half_ulp, dtype=np.float32)) == 0x3F800002,
            "subnormal_add_preserved": bits(np.add(minimum, minimum, dtype=np.float32)) == 2,
            "subnormal_multiply_preserved": bits(np.multiply(np.float32(2.0 ** -126), np.float32(0.5), dtype=np.float32)) == 0x00400000,
        }
    if not all(checks.values()):
        _fail("CPU_RN_OR_GRADUAL_UNDERFLOW_PRECHECK_FAILED", checks=checks)
    return checks


def _stage_float32(decoded, scale, row_ids):
    values = decoded.copy()
    with np.errstate(all="ignore"):
        for stage in range(8):
            stride = 1 << stage
            blocks = values.reshape(values.shape[0], -1, 2 * stride)
            low, high = blocks[..., :stride].copy(), blocks[..., stride:].copy()
            blocks[..., :stride] = np.add(low, high, dtype=np.float32)
            blocks[..., stride:] = np.subtract(low, high, dtype=np.float32)
            if not np.all(np.isfinite(values)):
                row, column = map(int, np.argwhere(~np.isfinite(values))[0])
                _fail("NONFINITE_FP32_STAGE", stage=stage, sample_row_index=row, row=row_ids[row], column=column,
                      cpu_pre_bits=_hex(values.view(np.uint32)[row, column], 8))
        values = np.multiply(values, np.float32(scale), dtype=np.float32)
    if not np.all(np.isfinite(values)):
        row, column = map(int, np.argwhere(~np.isfinite(values))[0])
        _fail("NONFINITE_FP32_SCALE", sample_row_index=row, row=row_ids[row], column=column,
              cpu_pre_bits=_hex(values.view(np.uint32)[row, column], 8))
    return values


def _exact_row(decoded_row, original_bits, dtype, scale):
    ratios = [float(value).as_integer_ratio() for value in decoded_row]
    for column, ((numerator, denominator), raw) in enumerate(zip(ratios, original_bits)):
        if Fraction(numerator, denominator) != _bits_fraction(int(raw), dtype):
            _fail("INPUT_DECODE_NOT_EXACT", column=column, input_bits=_hex(raw))
    common_denominator = max(denominator for _, denominator in ratios)
    integers = [numerator * (common_denominator // denominator) for numerator, denominator in ratios]
    l1 = Fraction(sum(abs(integer) for integer in integers), common_denominator)
    for stage in range(8):
        stride = 1 << stage
        for block in range(0, N, 2 * stride):
            for offset in range(stride):
                left, right = block + offset, block + offset + stride
                a, b = integers[left], integers[right]
                integers[left], integers[right] = a + b, a - b
    return [Fraction(integer, common_denominator) * scale for integer in integers], l1, common_denominator


@lru_cache(maxsize=1)
def _dense_matrix():
    result = np.array([[(-1.0 if (i & j).bit_count() % 2 else 1.0) for j in range(N)] for i in range(N)], dtype=np.float64)
    result.setflags(write=False)
    return result


def _neighbors(bits, dtype):
    exponent_bits, fraction_bits, _ = FORMATS[dtype]
    sign = 1 << (exponent_bits + fraction_bits)
    if bits & (sign - 1) == 0:
        lower, upper = sign | 1, 1
    elif bits & sign:
        lower, upper = bits + 1, bits - 1
    else:
        lower, upper = bits - 1, bits + 1
    current = _bits_fraction(bits, dtype)
    result = {}
    for name, neighbor in (("toward_negative", lower), ("toward_positive", upper)):
        result[name + "_bits"] = _hex(neighbor)
        try:
            distance = abs(_bits_fraction(neighbor, dtype) - current)
            result[name + "_ulp"] = _fraction_record(distance)
        except CertificateError:
            result[name + "_ulp"] = None
            result[name + "_neighbor_is_infinite"] = True
    return result


def _finite_round(value, dtype, *, negative_zero=False):
    bits = round_fraction_to_bits(value, dtype, negative_zero=negative_zero)
    return bits, _bits_fraction(bits, dtype)


def certify_samples(input_bits, gpu_output_bits, dtype, scale=1.0, *, dense_fp64=None,
                    row_ids=None, include_element_diagnostics=False):
    """Certify every element of caller-supplied sample rows; no arbitrary tolerances.

    dense_fp64: optional actual dense-oracle output, shape [sample_rows,256],
    dtype float64. Its deviation from exact arithmetic is reported, never added
    to the hard bound. If omitted, this function computes a NumPy FP64 dense
    product and records that provenance. All comparisons use Fraction exactly.
    """
    report = {"status": "FAIL", "certificate_version": "integer_fwht_fp32_storage_v2",
              "assumptions": ASSUMPTIONS, "gpu_executed": False, "numpy_version": np.__version__,
              "rows": [], "hard_gate_policy": "Exact rational E32; integer nearest-even storage bits must equal supplied GPU bits; no rounded-dense .01/.05 gate and no E64 added to E32."}
    try:
        if dtype not in ("fp16", "bf16"):
            _fail("INVALID_DTYPE")
        if isinstance(scale, (bool, np.bool_)) or not math.isfinite(float(scale)) or float(scale) not in (1.0, 0.0625):
            _fail("INVALID_SCALE")
        scale_fraction = Fraction(float(scale))
        inputs = _u16_matrix(input_bits, "input_bits")
        outputs = _u16_matrix(gpu_output_bits, "gpu_output_bits")
        if outputs.shape != inputs.shape:
            _fail("INPUT_OUTPUT_SHAPE_MISMATCH")
        count = inputs.shape[0]
        ids = list(range(count)) if row_ids is None else list(row_ids)
        if len(ids) != count or any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in ids):
            _fail("INVALID_ROW_IDS")
        ids = [int(value) for value in ids]
        _check_finite_bits(inputs, dtype, "input_bits", ids)
        _check_finite_bits(outputs, dtype, "gpu_output_bits", ids)
        report["cpu_environment_precheck"] = _check_cpu_environment()
        decoded = _decode(inputs, dtype)
        pre = _stage_float32(decoded, float(scale_fraction), ids)
        if dense_fp64 is None:
            dense = decoded.astype(np.float64) @ _dense_matrix() * np.float64(float(scale_fraction))
            report["dense_source"] = "computed_numpy_float64_dense_product"
        else:
            dense = np.asarray(dense_fp64)
            if dense.dtype != np.dtype(np.float64) or dense.shape != inputs.shape:
                _fail("INVALID_SUPPLIED_DENSE_FP64")
            dense = dense.copy()
            report["dense_source"] = "caller_supplied_actual_float64_dense_output"
        if not np.all(np.isfinite(dense)):
            row, column = map(int, np.argwhere(~np.isfinite(dense))[0])
            _fail("NONFINITE_DENSE_FP64", sample_row_index=row, row=ids[row], column=column)
        report.update(dtype=dtype, scale=_fraction_record(scale_fraction), sample_rows=count, elements=count * N,
                      input_u16_le_sha256=hashlib.sha256(inputs.astype("<u2").tobytes()).hexdigest(),
                      supplied_output_u16_le_sha256=hashlib.sha256(outputs.astype("<u2").tobytes()).hexdigest(),
                      cpu_pre_f32_le_sha256=hashlib.sha256(pre.astype("<f4").tobytes()).hexdigest(),
                      unit_roundoff_u32=_fraction_record(U32), eta32=_fraction_record(ETA32), gamma9=_fraction_record(GAMMA9))
        totals = {name: 0 for name in ("certified_elements", "dense_fp64_inexact_elements", "gpu_vs_exact_direct_storage_bit_differences",
                                      "exact_direct_vs_via_fp32_numeric_differences", "dense_direct_vs_via_fp32_numeric_differences",
                                      "signed_zero_only_rounding_differences")}
        global_maxima = {name: Fraction(0) for name in ("prestorage_error", "storage_rounding_error", "stored_error_vs_exact",
                           "dense_fp64_error_vs_exact", "stored_error_vs_unrounded_dense", "stored_error_vs_direct_rounded_dense",
                           "stored_error_vs_via_fp32_rounded_dense")}
        for row in range(count):
            try:
                exact_values, l1, denominator = _exact_row(decoded[row], inputs[row], dtype, scale_fraction)
            except CertificateError as error:
                error.failure.update(sample_row_index=row, row=ids[row])
                raise
            relative_bound = GAMMA9 * abs(scale_fraction) * l1
            underflow_bound = (ETA32 / 2) * (abs(scale_fraction) * (N - 1) * (1 + U32) ** 8 + 1)
            bound = relative_bound + underflow_bound
            row_maxima = {name: Fraction(0) for name in global_maxima}
            row_counts = {name: 0 for name in totals}
            record = {"sample_row_index": row, "row": ids[row], "common_input_denominator": str(denominator),
                      "input_l1": _fraction_record(l1), "relative_pre_bound": _fraction_record(relative_bound),
                      "underflow_pre_bound": _fraction_record(underflow_bound), "pre_bound": _fraction_record(bound),
                      "diagnostic_examples": []}
            if include_element_diagnostics:
                record["elements"] = []
            worst = None
            for column, exact in enumerate(exact_values):
                cpu_value = Fraction(*float(pre[row, column]).as_integer_ratio())
                gpu_bits = int(outputs[row, column])
                expected_bits = round_fraction_to_bits(cpu_value, dtype, negative_zero=bool(np.signbit(pre[row, column])))
                context = {"sample_row_index": row, "row": ids[row], "column": column,
                           "gpu_bits": _hex(gpu_bits), "expected_storage_bits": _hex(expected_bits),
                           "cpu_pre_bits": _hex(pre.view(np.uint32)[row, column], 8)}
                try:
                    expected_stored = _bits_fraction(expected_bits, dtype)
                except CertificateError:
                    _fail("STORAGE_CONVERSION_OVERFLOW", **context)
                if gpu_bits != expected_bits:
                    _fail("GPU_VS_INTEGER_STORAGE_RNE_BITS", **context)
                gpu = _bits_fraction(gpu_bits, dtype)
                pre_error = abs(cpu_value - exact)
                delta_store = abs(expected_stored - cpu_value)
                stored_error = abs(gpu - exact)
                if pre_error > bound:
                    _fail("PRESTORAGE_BOUND_EXCEEDED", **context, exact_value=str(exact), cpu_pre_value=str(cpu_value),
                          actual_error=str(pre_error), bound=str(bound))
                if stored_error > bound + delta_store:
                    _fail("STORED_BOUND_EXCEEDED", **context, exact_value=str(exact), actual_error=str(stored_error),
                          pre_bound=str(bound), actual_storage_rounding_error=str(delta_store))
                dense_value = Fraction(*float(dense[row, column]).as_integer_ratio())
                try:
                    direct_bits, direct = _finite_round(exact, dtype)
                    exact32_bits, exact32 = _finite_round(exact, "fp32")
                    via_bits, via = _finite_round(exact32, dtype, negative_zero=bool(exact32_bits >> 31))
                    dense_direct_bits, dense_direct = _finite_round(dense_value, dtype, negative_zero=bool(np.signbit(dense[row, column])))
                    dense32_bits, dense32 = _finite_round(dense_value, "fp32", negative_zero=bool(np.signbit(dense[row, column])))
                    dense_via_bits, dense_via = _finite_round(dense32, dtype, negative_zero=bool(dense32_bits >> 31))
                except CertificateError:
                    _fail("NONFINITE_REFERENCE_ROUNDING_DIAGNOSTIC", **context)
                metrics = {"prestorage_error": pre_error, "storage_rounding_error": delta_store,
                           "stored_error_vs_exact": stored_error, "dense_fp64_error_vs_exact": abs(dense_value - exact),
                           "stored_error_vs_unrounded_dense": abs(gpu - dense_value),
                           "stored_error_vs_direct_rounded_dense": abs(gpu - dense_direct),
                           "stored_error_vs_via_fp32_rounded_dense": abs(gpu - dense_via)}
                for name, value in metrics.items():
                    row_maxima[name] = max(row_maxima[name], value)
                    global_maxima[name] = max(global_maxima[name], value)
                flags = {"certified_elements": 1, "dense_fp64_inexact_elements": int(dense_value != exact),
                         "gpu_vs_exact_direct_storage_bit_differences": int(gpu_bits != direct_bits),
                         "exact_direct_vs_via_fp32_numeric_differences": int(direct != via),
                         "dense_direct_vs_via_fp32_numeric_differences": int(dense_direct != dense_via),
                         "signed_zero_only_rounding_differences": int((gpu_bits != direct_bits and gpu == direct) or
                                                                     (direct_bits != via_bits and direct == via) or
                                                                     (dense_direct_bits != dense_via_bits and dense_direct == dense_via))}
                for name, value in flags.items():
                    row_counts[name] += value
                    totals[name] += value
                new_worst = worst is None or stored_error > worst[0]
                keep_example = (gpu_bits != direct_bits or direct_bits != via_bits or dense_direct_bits != dense_via_bits) and len(record["diagnostic_examples"]) < 8
                if include_element_diagnostics or new_worst or keep_example:
                    element = {**context, "exact_value": str(exact), "cpu_pre_value": str(cpu_value), "gpu_stored_value": str(gpu),
                               "dense_fp64_value": str(dense_value), "exact_direct_storage_bits": _hex(direct_bits),
                               "exact_via_fp32_storage_bits": _hex(via_bits), "dense_direct_storage_bits": _hex(dense_direct_bits),
                               "dense_via_fp32_storage_bits": _hex(dense_via_bits),
                               "errors": {name: _fraction_record(value) for name, value in metrics.items()},
                               "stored_bound": _fraction_record(bound + delta_store), "neighbors": _neighbors(gpu_bits, dtype)}
                if new_worst:
                    worst = (stored_error, element)
                if keep_example:
                    record["diagnostic_examples"].append(element)
                if include_element_diagnostics:
                    record["elements"].append(element)
            record.update(status="PASS", counts=row_counts, maxima={name: _fraction_record(value) for name, value in row_maxima.items()},
                          worst_stored_error_element=worst[1], dense_fp64_is_exact=row_counts["dense_fp64_inexact_elements"] == 0,
                          all_three_hard_gates_passed=True, diagnostic_examples_limit=8)
            report["rows"].append(record)
        report.update(status="PASS", summary={"counts": totals, "maxima": {name: _fraction_record(value) for name, value in global_maxima.items()},
                      "all_three_hard_gates_passed": True, "dense_fp64_error_is_diagnostic_only": True}, first_failure=None)
    except CertificateError as error:
        report["first_failure"] = error.failure
    except (ValueError, TypeError, OverflowError, ArithmeticError) as error:
        report["first_failure"] = {"code": "INVALID_INPUT_OR_CPU_ARITHMETIC", "exception": type(error).__name__, "message": str(error)}
    return report
