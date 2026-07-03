# CUDA Samples particles: DEM collision over a uniform grid

Discrete-element sphere-sphere collision (spring + dashpot + shear forces)
resolved through a spatially hashed uniform grid with cellStart/cellEnd
bucket traversal — the core physics kernel of the classic CUDA "particles"
sample. Exercises a __constant__ parameter struct, float4 data, 27-cell
neighborhood loops, and indirect writes through a sorted-index map.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh

Upstream commit: b7c5481c (snapshot: benchmark/collection/molecular-dynamics/sources/md-07)

License: BSD-style (NVIDIA)

Extraction fidelity: extracted

Extraction notes: calcGridPos, calcGridHash, collideSpheres, collideCell and
collideD are upstream device code verbatim, with cudaParams kept as the
__constant__ block (subset of SimParams). Upstream sorts particles with
thrust and a reorder kernel; the harness prebuilds identical sorted arrays
plus cellStart/cellEnd deterministically on the host (counting sort), so
the case isolates the collision kernel. float3/float4 operators are the
required subset of cuda-samples helper_math.h. The CPU oracle uses an
O(n^2) cutoff sum, which is exactly equivalent because cellSize equals the
collision distance.
