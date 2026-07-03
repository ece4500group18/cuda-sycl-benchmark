# nvCOMP batched CRC32 checksums

PKZIP CRC32 checksums of batched data chunks computed on the GPU via nvCOMP.

Source project: NVIDIA/nvcomp

Source URL: https://github.com/NVIDIA/nvcomp/blob/main/examples/nvcomp_crc32.cu

License: Apache-2.0

Extraction fidelity: extracted

Extraction notes: nvcompBatchedCRC32GetHeuristicConf + nvcompBatchedCRC32Async sequence and CPU reference helpers verbatim; deterministic hash chunks replace file input and BatchData helpers. Snapshot lib-06.
