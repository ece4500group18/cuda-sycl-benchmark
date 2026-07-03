# Rodinia k-means device/host iteration

K-means iteration: transpose features, assign nearest centroids on device, recompute centroids on host.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/kmeans-cuda/cluster.cu

Upstream commit: 01f58fc5

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: feature_transpose + find_membership kernels verbatim; host recomputes centroids (double accumulation) between launches; fixed 4 iterations, centroids seeded with first points. From the HeCBench raw clone (not yet a collection snapshot).
