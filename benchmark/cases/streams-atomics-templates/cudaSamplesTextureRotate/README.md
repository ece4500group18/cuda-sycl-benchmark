# Texture-object image rotation (linear filter, wrap addressing)

Rotate an image by sampling a texture object with normalized coordinates, bilinear filtering and wrap addressing.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/simpleTexture/simpleTexture.cu

Upstream commit: b7c5481c

License: BSD-3-Clause + CUDA EULA note

Extraction fidelity: extracted

Extraction notes: transformKernel and the full texture-object setup (channel desc, cudaArray, resource/texture descriptors, cudaCreateTextureObject) verbatim; deterministic smooth image replaces the PGM input; tolerance covers the 9-bit interpolation-weight quantization.
