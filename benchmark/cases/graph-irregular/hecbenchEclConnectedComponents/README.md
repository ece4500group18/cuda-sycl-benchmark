# ECL-CC connected components (5-kernel)

Connected components via lock-free hooking and pointer jumping with degree-partitioned worklists.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/cc-cuda/main.cu

Upstream commit: 01f58fc5

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: init/compute1/compute2/compute3/flatten + representative() verbatim incl. device worklist counters, atomicCAS hooking, warp shuffle and __launch_bounds__. Snapshot graph-06.
