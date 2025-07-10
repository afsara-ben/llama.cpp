from __future__ import annotations

"""-------------------------------------------------------------------------
Latency predictor (tiny roofline-based).
Now integrates the sample throughput numbers you provided for M2 Ultra
    • FP16 4096×14336  → 623 GFLOP/s
    • IQ1_M            → 1.61 TFLOP/s  (1610 GFLOP/s)
    • Q2_K             → 2.56 TFLOP/s  (2560 GFLOP/s)
Feel free to extend the REF_EFF dict with the rest of the table.
-------------------------------------------------------------------------"""

# from dataclasses import dataclass
# from typing import Dict, List

# # ---------------------------------------------------------------------
# # Data classes
# # ---------------------------------------------------------------------

# @dataclass
# class HardwareSpec:
#     name: str
#     peak_gflops_by_bit: Dict[int, float]  # {bit‑width: peak‑GFLOP/s}
#     mem_bw_gbs: float                     # sustained DRAM GB/s
#     launch_overhead_us: float = 4.0

#     def peak_compute(self, bit_category: int) -> float:
#         if bit_category not in self.peak_gflops_by_bit:
#             larger = sorted(k for k in self.peak_gflops_by_bit if k > bit_category)
#             if not larger:
#                 raise KeyError(f"No peak entry for ≤{bit_category}-bit ops on {self.name}")
#             bit_category = larger[0]
#         return self.peak_gflops_by_bit[bit_category]


# @dataclass
# class QuantType:
#     name: str
#     stored_bits: float
#     compute_path_bits: int
#     extra_bytes_per_elem: float = 0.0  # tweak for codebook / scale tables


# @dataclass
# class MatVecLayer:
#     name: str
#     rows: int
#     cols: int
#     quant: QuantType

#     @property
#     def flops(self) -> float:
#         return 2.0 * self.rows * self.cols

#     @property
#     def elements(self) -> int:
#         return self.rows * self.cols

#     def bytes_moved(self) -> float:
#         weight_bytes = self.elements * self.quant.stored_bits / 8.0
#         return weight_bytes + self.elements * self.quant.extra_bytes_per_elem


# # ---------------------------------------------------------------------
# # Predictor
# # ---------------------------------------------------------------------

# class LatencyPredictor:
#     def __init__(self, ref_hw: HardwareSpec, ref_eff: Dict[str, float]):
#         self.ref_hw = ref_hw
#         self.ref_eff = ref_eff

#     def predict_gflops(self, quant: QuantType, target_hw: HardwareSpec) -> float:
#         ref_peak = self.ref_hw.peak_compute(quant.compute_path_bits)
#         tgt_peak = target_hw.peak_compute(quant.compute_path_bits)
#         eff = self.ref_eff[quant.name] / ref_peak  # efficiency 0‑1
#         return eff * tgt_peak

#     def kernel_latency_ms(self, layer: MatVecLayer, target_hw: HardwareSpec) -> float:
#         gflops = self.predict_gflops(layer.quant, target_hw)
#         compute_ms = layer.flops / (gflops * 1e6)
#         mem_ms = layer.bytes_moved() / (target_hw.mem_bw_gbs * 1e9) * 1e3
#         return max(compute_ms, mem_ms) + target_hw.launch_overhead_us / 1000.0

#     def sequence_latency_ms(self, layers: List[MatVecLayer], target_hw: HardwareSpec) -> float:
#         return sum(self.kernel_latency_ms(ly, target_hw) for ly in layers)


# # ---------------------------------------------------------------------
# # Quick demo using your sample numbers
# # ---------------------------------------------------------------------

# if __name__ == "__main__":
#     # Hardware
#     m2_ultra = HardwareSpec(
#         "M2 Ultra",
#         peak_gflops_by_bit={32: 60000, 16: 120000, 8: 240000},
#         mem_bw_gbs=800,
#     )
#     m2_max = HardwareSpec(
#         "M2 Max",
#         peak_gflops_by_bit={32: 30000, 16: 60000, 8: 120000},
#         mem_bw_gbs=400,
#     )

#     # Quant types
#     FP16 = QuantType("FP16", stored_bits=16, compute_path_bits=16)
#     IQ1_M = QuantType("IQ1_M", stored_bits=1.75, compute_path_bits=8, extra_bytes_per_elem=0.5)
#     Q2_K  = QuantType("Q2_K",  stored_bits=2.95, compute_path_bits=8, extra_bytes_per_elem=0.4)

#     # Layers (decode‑phase example)
#     layers = [
#         MatVecLayer("Q_proj", rows=4096, cols=14336, quant=IQ1_M),
#         MatVecLayer("V_proj", rows=4096, cols=14336, quant=Q2_K),
#     ]

#     # Reference efficiencies from your table (GFLOP/s on M2 Ultra)
#     REF_EFF = {
#         "FP16": 623.0,   # 0.623 TFLOP/s for the 4096×14336 gemv
#         "IQ1_M": 1610.0, # 1.61 TFLOP/s
#         "Q2_K": 2560.0,  # 2.56 TFLOP/s
#     }

#     predictor = LatencyPredictor(m2_ultra, REF_EFF)

#     for gpu in (m2_ultra, m2_max):
#         print(f"\nPer‑kernel latency on {gpu.name} (ms):")
#         for ly in layers:
#             print(f"  {ly.name:8}: {predictor.kernel_latency_ms(ly, gpu):.3f}")
#         print(f"Total decode latency: {predictor.sequence_latency_ms(layers, gpu):.3f} ms")






"""-------------------------------------------------------------------------
GPU‑Kernel Performance Toolkit — v0.3.2 (parsing‑fix release)

Fixes
-----
• **ValueError** when parsing bit‑width from kernel names like `q6_K`.
  We now extract the *first integer substring* anywhere in the name.
• Minor: cleaned up demo print formatting.

Quick use
---------
Run this file: it trains on the n=1 & n=8 data (M2 Ultra) and prints
n=2 GFLOP/s predictions for both M2 Ultra and M2 Max.
-------------------------------------------------------------------------"""

# from dataclasses import dataclass
# from typing import Dict, List, Tuple
# import numpy as np
# import re

# # ~~~~~~~~~~~~~~~~~~~~~~~~~ Hardware & quant metadata ~~~~~~~~~~~~~~~~~~~~~~~~~

# @dataclass
# class HardwareSpec:
#     name: str
#     peak_gflops_by_bit: Dict[int, float]
#     mem_bw_gbs: float
#     launch_overhead_us: float = 4.0

#     def peak_compute(self, bit_cat: int) -> float:
#         if bit_cat in self.peak_gflops_by_bit:
#             return self.peak_gflops_by_bit[bit_cat]
#         larger = sorted(k for k in self.peak_gflops_by_bit if k > bit_cat)
#         if not larger:
#             raise KeyError(f"{self.name}: no peak entry ≥{bit_cat}-bit")
#         return self.peak_gflops_by_bit[larger[0]]


# @dataclass
# class QuantType:
#     name: str
#     stored_bits: float
#     compute_path_bits: int


# # ~~~~~~~~~~~~~~~~~~~~~~~~~ Instruction statistics ~~~~~~~~~~~~~~~~~~~~~~~~~~~

# STAT_KEYS = ["ALU", "TOTAL_INT", "Wait", "Device Load"]

# @dataclass
# class InstructionStat:
#     counts: Dict[str, int]

#     def as_ratios(self) -> List[float]:
#         total = sum(self.counts.values()) or 1
#         return [self.counts.get(k, 0) / total for k in STAT_KEYS]


# # ~~~~~~~~~~~~~~~~~~~~~~~~~ Quadratic Ridge predictor ~~~~~~~~~~~~~~~~~~~~~~~~

# class QuadraticRidgePredictor:
#     """Ridge‑regularised quadratic regressor on efficiency."""

#     def __init__(
#         self,
#         src_hw: HardwareSpec,
#         rows: List[Tuple[str, int, float]],  # (kernel, n, gflops)
#         qt_map: Dict[str, QuantType],
#         stats: Dict[str, InstructionStat],
#         alpha: float = 1e-2,
#     ):
#         self.src_hw, self.qt_map, self.stats = src_hw, qt_map, stats
#         self.kernels = sorted({k for k, _, _ in rows})
#         zstat = {k: 0 for k in STAT_KEYS}
#         self.default_stat = InstructionStat(zstat)
#         self.alpha = alpha
#         self._fit(rows)

#     # ------------------- helpers -------------------
#     def _encode(self, kernel: str, n: int) -> np.ndarray:
#         qt = self.qt_map[kernel]
#         st = self.stats.get(kernel, self.default_stat)
#         base = [n, n ** 2, qt.stored_bits, *st.as_ratios()]
#         oh = np.zeros(len(self.kernels))
#         oh[self.kernels.index(kernel)] = 1.0
#         return np.concatenate([base, oh])

#     def _fit(self, rows):
#         X = np.vstack([self._encode(k, n) for k, n, _ in rows])
#         y = np.array([
#             g / self.src_hw.peak_compute(self.qt_map[k].compute_path_bits)
#             for k, n, g in rows
#         ])
#         regI = self.alpha * np.eye(X.shape[1])
#         self.coef_ = np.linalg.pinv(X.T @ X + regI) @ X.T @ y

#     # ------------------- API -------------------
#     def predict_eff(self, kernel: str, n: int) -> float:
#         return float(self._encode(kernel, n) @ self.coef_)

#     def predict_gflops(self, kernel: str, n: int, tgt_hw: HardwareSpec) -> float:
#         eff = np.clip(self.predict_eff(kernel, n), 0, 1)
#         return eff * tgt_hw.peak_compute(self.qt_map[kernel].compute_path_bits)


# # ~~~~~~~~~~~~~~~~~~~~~~~~~ Helpers for demo ~~~~~~~~~~~~~~~~~~~~~~~~~

# def extract_bits(name: str, default: int = 8) -> int:
#     """Return the first integer substring found in the kernel name."""
#     m = re.search(r"(\d+)", name)
#     return int(m.group(1)) if m else default


# def make_quant_catalog(names: List[str]) -> Dict[str, QuantType]:
#     def q(name, bits):
#         return QuantType(name, bits, 8)
#     catalog: Dict[str, QuantType] = {}
#     for nm in names:
#         catalog[nm] = q(nm, extract_bits(nm))
#     # bespoke entries
#     catalog["iq1_m"] = QuantType("iq1_m", 1.75, 8)
#     return catalog


# # ~~~~~~~~~~~~~~~~~~~~~~~~~ Demo section ~~~~~~~~~~~~~~~~~~~~~~~~~
# if __name__ == "__main__":
#     ultra = HardwareSpec("M2 Ultra", {32: 60_000, 16: 120_000, 8: 240_000}, 800)
#     mmax = HardwareSpec("M2 Max", {32: 30_000, 16: 60_000, 8: 120_000}, 400)

#     kernel_names = [
#         "q6_K", "q4_0", "q5_K", "q4_K", "q2_K", "q3_K",
#         "iq2_xxs", "iq3_s",
#     ]
#     QT = make_quant_catalog(kernel_names)

#     # Instruction counts subset (extend as needed)
#     COUNT = {
#         "q6_K": {"ALU": 278, "TOTAL_INT": 144, "Wait": 2, "Device Load": 12},
#         "q4_0": {"ALU": 453, "TOTAL_INT": 149, "Wait": 2, "Device Load": 12},
#         "q5_K": {"ALU": 677, "TOTAL_INT": 193, "Wait": 4, "Device Load": 28},
#         "iq1_m": {"ALU": 393, "TOTAL_INT": 176, "Wait": 5, "Device Load": 16},
#         "iq2_xxs": {"ALU": 466, "TOTAL_INT": 256, "Wait": 4, "Device Load": 8},
#         "iq3_s": {"ALU": 463, "TOTAL_INT": 211, "Wait": 3, "Device Load": 16},
#         "q4_K": {"ALU": 509, "TOTAL_INT": 138, "Wait": 4, "Device Load": 20},
#         "q2_K": {"ALU": 508, "TOTAL_INT": 134, "Wait": 4, "Device Load": 20},
#         "q3_K": {"ALU": 653, "TOTAL_INT": 204, "Wait": 3, "Device Load": 20},
#     }
#     INSTR = {k: InstructionStat(v) for k, v in COUNT.items()}

#     # Timing data (µs)
#     RAW_N1 = {
#         "q6_K": 75.67, "q4_0": 38.8, "q5_K": 64.49, "iq1_m": 72.99,
#         "iq2_xxs": 70.13, "iq3_s": 71.17, "q4_K": 49.81, "q2_K": 45.9,
#         "q3_K": 57.59,
#     }
#     RAW_N8 = {
#         "q6_K": 364.93, "q4_0": 255.74, "q5_K": 401.56, "iq1_m": 630.01,
#         "iq2_xxs": 675.42, "iq3_s": 669.77, "q4_K": 372.87, "q2_K": 555.59,
#         "q3_K": 648.91,
#     }

#     MFLOP1 = 2 * 4096 * 14336 / 1e6
#     MFLOP8 = MFLOP1 * 8
#     data_rows = [(k, 1, MFLOP1 / (t * 1e-6)) for k, t in RAW_N1.items()] + [
#         (k, 8, MFLOP8 / (t * 1e-6)) for k, t in RAW_N8.items()
#     ]

#     predictor = QuadraticRidgePredictor(ultra, data_rows, QT, INSTR)

#     for gpu in (ultra, mmax):
#         print(f"\n[Quadratic] Predicted GFLOP/s on {gpu.name} (n=2):")
#         for k in sorted(QT):
#             g = predictor.predict_gflops(k, 2, gpu) / 1e3
#             print(f"  {k:10}: {g:5.2f} TFLOP/s")



"""-------------------------------------------------------------------------
GPU‑Kernel Performance Toolkit — v0.3.2 (parsing‑fix release)

Fixes
-----
• **ValueError** when parsing bit‑width from kernel names like `q6_K`.
  We now extract the *first integer substring* anywhere in the name.
• Minor: cleaned up demo print formatting.

Quick use
---------
Run this file: it trains on the n=1 & n=8 data (M2 Ultra) and prints
n=2 GFLOP/s predictions for both M2 Ultra and M2 Max.
-------------------------------------------------------------------------"""

# from dataclasses import dataclass
# from typing import Dict, List, Tuple
# import numpy as np
# import re

# # ~~~~~~~~~~~~~~~~~~~~~~~~~ Hardware & quant metadata ~~~~~~~~~~~~~~~~~~~~~~~~~

# @dataclass
# class HardwareSpec:
#     name: str
#     peak_gflops_by_bit: Dict[int, float]
#     mem_bw_gbs: float
#     launch_overhead_us: float = 4.0

#     def peak_compute(self, bit_cat: int) -> float:
#         if bit_cat in self.peak_gflops_by_bit:
#             return self.peak_gflops_by_bit[bit_cat]
#         larger = sorted(k for k in self.peak_gflops_by_bit if k > bit_cat)
#         if not larger:
#             raise KeyError(f"{self.name}: no peak entry ≥{bit_cat}-bit")
#         return self.peak_gflops_by_bit[larger[0]]


# @dataclass
# class QuantType:
#     name: str
#     stored_bits: float
#     compute_path_bits: int


# # ~~~~~~~~~~~~~~~~~~~~~~~~~ Instruction statistics ~~~~~~~~~~~~~~~~~~~~~~~~~~~

# STAT_KEYS = ["ALU", "TOTAL_INT", "Wait", "Device Load"]

# @dataclass
# class InstructionStat:
#     counts: Dict[str, int]

#     def as_ratios(self) -> List[float]:
#         total = sum(self.counts.values()) or 1
#         return [self.counts.get(k, 0) / total for k in STAT_KEYS]


# # ~~~~~~~~~~~~~~~~~~~~~~~~~ Per‑kernel linear interpolator ~~~~~~~~~~~~~~~~~~~~

# class InterpPredictor:
#     """Predicts efficiency for any n by linear interpolation between the two
#     anchor points (n=1 and n=8) recorded for that kernel.

#     If you have more anchor points you can switch to numpy.polyfit or splines
#     easily. For now we keep it minimal and robust.
#     """

#     def __init__(
#         self,
#         src_hw: HardwareSpec,
#         anchor: Dict[str, Dict[int, float]],  # {kernel → {n_val: gflops}}
#         qt_map: Dict[str, QuantType],
#     ):
#         self.src_hw = src_hw
#         self.anchor = anchor
#         self.qt_map = qt_map

#     def _eff(self, kernel: str, n: int) -> float:
#         data = self.anchor[kernel]
#         ns = sorted(data)
#         if n in data:
#             return data[n] / self.src_hw.peak_compute(self.qt_map[kernel].compute_path_bits)
#         # linear between nearest neighbours
#         for i in range(len(ns) - 1):
#             if ns[i] < n < ns[i + 1]:
#                 n0, n1 = ns[i], ns[i + 1]
#                 g0, g1 = data[n0], data[n1]
#                 slope = (g1 - g0) / (n1 - n0)
#                 g = g0 + slope * (n - n0)
#                 return g / self.src_hw.peak_compute(self.qt_map[kernel].compute_path_bits)
#         # extrapolate if outside range
#         n0, n1 = ns[0], ns[-1]
#         g0, g1 = data[n0], data[n1]
#         slope = (g1 - g0) / (n1 - n0)
#         if n < n0:
#             g = g0 + slope * (n - n0)
#         else:
#             g = g1 + slope * (n - n1)
#         return g / self.src_hw.peak_compute(self.qt_map[kernel].compute_path_bits)

#     def predict_gflops(self, kernel: str, n: int, tgt_hw: HardwareSpec) -> float:
#         eff = self._eff(kernel, n)
#         return eff * tgt_hw.peak_compute(self.qt_map[kernel].compute_path_bits)

# def extract_bits(name: str, default: int = 8) -> int:
#     """Return the first integer substring found in the kernel name."""
#     m = re.search(r"(\d+)", name)
#     return int(m.group(1)) if m else default


# def make_quant_catalog(names: List[str]) -> Dict[str, QuantType]:
#     def q(name, bits):
#         return QuantType(name, bits, 8)
#     catalog: Dict[str, QuantType] = {}
#     for nm in names:
#         catalog[nm] = q(nm, extract_bits(nm))
#     # bespoke entries
#     catalog["iq1_m"] = QuantType("iq1_m", 1.75, 8)
#     return catalog


# # ~~~~~~~~~~~~~~~~~~~~~~~~~ Demo section ~~~~~~~~~~~~~~~~~~~~~~~~~
# if __name__ == "__main__":
#     ultra = HardwareSpec("M2 Ultra", {32: 60_000, 16: 120_000, 8: 240_000}, 800)
#     mmax = HardwareSpec("M2 Max", {32: 30_000, 16: 60_000, 8: 120_000}, 400)

#     kernel_names = [
#         "q6_K", "q4_0", "q5_K", "q4_K", "q2_K", "q3_K",
#         "iq2_xxs", "iq3_s",
#     ]
#     QT = make_quant_catalog(kernel_names)

#     # Instruction counts subset (extend as needed)
#     COUNT = {
#         "q6_K": {"ALU": 278, "TOTAL_INT": 144, "Wait": 2, "Device Load": 12},
#         "q4_0": {"ALU": 453, "TOTAL_INT": 149, "Wait": 2, "Device Load": 12},
#         "q5_K": {"ALU": 677, "TOTAL_INT": 193, "Wait": 4, "Device Load": 28},
#         "iq1_m": {"ALU": 393, "TOTAL_INT": 176, "Wait": 5, "Device Load": 16},
#         "iq2_xxs": {"ALU": 466, "TOTAL_INT": 256, "Wait": 4, "Device Load": 8},
#         "iq3_s": {"ALU": 463, "TOTAL_INT": 211, "Wait": 3, "Device Load": 16},
#         "q4_K": {"ALU": 509, "TOTAL_INT": 138, "Wait": 4, "Device Load": 20},
#         "q2_K": {"ALU": 508, "TOTAL_INT": 134, "Wait": 4, "Device Load": 20},
#         "q3_K": {"ALU": 653, "TOTAL_INT": 204, "Wait": 3, "Device Load": 20},
#     }
#     INSTR = {k: InstructionStat(v) for k, v in COUNT.items()}

#     # Timing data (µs)
#     RAW_N1 = {
#         "q6_K": 75.67, "q4_0": 38.8, "q5_K": 64.49, "iq1_m": 72.99,
#         "iq2_xxs": 70.13, "iq3_s": 71.17, "q4_K": 49.81, "q2_K": 45.9,
#         "q3_K": 57.59,
#     }
#     RAW_N8 = {
#         "q6_K": 364.93, "q4_0": 255.74, "q5_K": 401.56, "iq1_m": 630.01,
#         "iq2_xxs": 675.42, "iq3_s": 669.77, "q4_K": 372.87, "q2_K": 555.59,
#         "q3_K": 648.91,
#     }

#     MFLOP1 = 2 * 4096 * 14336 / 1e6
#     MFLOP8 = MFLOP1 * 8
#     data_rows = [(k, 1, MFLOP1 / (t * 1e-6)) for k, t in RAW_N1.items()] + [
#         (k, 8, MFLOP8 / (t * 1e-6)) for k, t in RAW_N8.items()
#     ]

#         # Build anchor dict for interpolation
#     anchor: Dict[str, Dict[int, float]] = {}
#     for k in RAW_N1:
#         g1 = MFLOP1 / (RAW_N1[k] * 1e-6)
#         g8 = MFLOP8 / (RAW_N8[k] * 1e-6)
#         anchor[k] = {1: g1, 8: g8}

#     predictor = InterpPredictor(ultra, anchor, QT)

#     for gpu in (ultra, mmax):
#         print(f"[Interp] Predicted GFLOP/s on {gpu.name} (n=2):")
#         for k in sorted(QT):
#             g = predictor.predict_gflops(k, 2, gpu) / 1e3
#             print(f"  {k:10}: {g:5.2f} TFLOP/s {ultra, mmax}")
#             print(f"\n[Quadratic] Predicted GFLOP/s on {gpu.name} (n=2):")
#             for k in sorted(QT):
#                 g = predictor.predict_gflops(k, 2, gpu) / 1e3
#                 print(f"  {k:10}: {g:5.2f} TFLOP/s")



"""-------------------------------------------------------------------------
GPU‑Kernel Performance Toolkit — *clean* interpolation‑only edition

Changes in this revision (2025‑07‑09)
------------------------------------
1. **Removed FP16 path** — we treat FP16 the same as FP32, so only FP32
   (`32‑bit`) and INT8 (`8‑bit`) peaks are kept in the `HardwareSpec`.
2. **Fixed broken merge artefacts** that produced a syntax error and
   duplicate print loops.
3. **De‑quant overhead note** – each `QuantType` still carries
   `stored_bits`; any unpack/dequant cost is implicitly captured by the
   empirical GFLOP/s anchors at n=1 and n=8, so no extra modelling is
   required for the interpolation baseline.
-------------------------------------------------------------------------"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import re

# ~~~~~~~~~~~~~~~~~~~~~~~~~ Hardware & quant metadata ~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class HardwareSpec:
    name: str
    peak_gflops_by_bit: Dict[int, float]  # e.g. {32: 27_200, 8: 240_000}
    mem_bw_gbs: float
    launch_overhead_us: float = 4.0

    def peak_compute(self, bit_cat: int) -> float:
        if bit_cat in self.peak_gflops_by_bit:
            return self.peak_gflops_by_bit[bit_cat]
        raise KeyError(f"{self.name}: no peak entry for {bit_cat}-bit path")


@dataclass
class QuantType:
    name: str
    stored_bits: float
    compute_path_bits: int = 8  # All sub‑8‑bit kernels map to INT8 path.


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Simple interpolation predictor ~~~~~~~~~~~~~~~~~~~~

class InterpPredictor:
    """Per‑kernel linear interpolation between anchor points (n=1, n=8)."""

    def __init__(self, src_hw: HardwareSpec, anchor: Dict[str, Dict[int, float]], qt_map: Dict[str, QuantType]):
        self.src_hw = src_hw
        self.anchor = anchor
        self.qt_map = qt_map

    def _eff(self, kernel: str, n: int) -> float:
        data = self.anchor[kernel]
        if n in data:
            return data[n] / self.src_hw.peak_compute(self.qt_map[kernel].compute_path_bits)
        n0, n1 = sorted(data)  # assumes exactly two anchors 1 and 8
        g0, g1 = data[n0], data[n1]
        g = g0 + (g1 - g0) * (n - n0) / (n1 - n0)
        return g / self.src_hw.peak_compute(self.qt_map[kernel].compute_path_bits)

    def predict_gflops(self, kernel: str, n: int, tgt_hw: HardwareSpec) -> float:
        eff = self._eff(kernel, n)
        return eff * tgt_hw.peak_compute(self.qt_map[kernel].compute_path_bits)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Helper utilities ~~~~~~~~~~~~~~~~~~~~~~~~~

def extract_bits(name: str, default: int = 8) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else default


def make_quant_catalog(names: List[str]) -> Dict[str, QuantType]:
    return {nm: QuantType(nm, extract_bits(nm)) for nm in names}


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Demo / self‑test ~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    # Hardware specs (FP16 removed – treated same as FP32)
    ultra = HardwareSpec("M2 Ultra", {32: 27_200, 8: 240_000}, mem_bw_gbs=800)
    mmax  = HardwareSpec("M2 Max",   {32: 30_000, 8: 120_000}, mem_bw_gbs=400)

    kernels = [
        "q2_K", "q3_K", "q4_0", "q4_K", "q5_K", "q6_K",
        "iq1_m", "iq2_xxs", "iq3_s",
    ]
    QT = make_quant_catalog(kernels)

    # Anchors: GFLOP/s measured on M2 Ultra at n=1 and n=8
    RAW_US_N1 = {
        "q6_K": 75.67, "q4_0": 38.8, "q5_K": 64.49, "iq1_m": 72.99,
        "iq2_xxs": 70.13, "iq3_s": 71.17, "q4_K": 49.81, "q2_K": 45.9,
        "q3_K": 57.59,
    }
    RAW_US_N8 = {
        "q6_K": 364.93, "q4_0": 255.74, "q5_K": 401.56, "iq1_m": 630.01,
        "iq2_xxs": 675.42, "iq3_s": 669.77, "q4_K": 372.87, "q2_K": 555.59,
        "q3_K": 648.91,
    }

    MFLOP_per_n = 2 * 4096 * 14336 / 1e6  # 117.44 MFLOP per n=1

    anchor: Dict[str, Dict[int, float]] = {}
    for k in RAW_US_N1:
        g1 = MFLOP_per_n / (RAW_US_N1[k] * 1e-6)
        g8 = MFLOP_per_n * 8 / (RAW_US_N8[k] * 1e-6)
        anchor[k] = {1: g1, 8: g8}

    predictor = InterpPredictor(ultra, anchor, QT)

    for gpu in (ultra, mmax):
        print(f"\n[Interp] Predicted GFLOP/s on {gpu.name} for n = 2 (m=4096, k=14336):")
        for k in sorted(kernels):
            g_tflops = predictor.predict_gflops(k, 2, gpu) / 1e3
            print(f"  {k:10}: {g_tflops:5.2f} GFLOP/s")
