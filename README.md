Torch FFI-bindings for NNPACK 
=============

NNPACK is a fast CPU implementation of convolution operations for training ConvNets:
 
https://github.com/Maratyszcza/NNPACK

**This is a work in progress**

Limitations of NNPACK:

 * strided convolutions are not supported
 * there is no scale parameter on `accGradParameters` call 

Limitations of these bindings:

 * no pthreadpool yet, runs in the same thread
 * `accGradParameters` test is failing on `gradWeight` and `gradBias`

# Installation

Follow installation steps at https://github.com/Maratyszcza/NNPACK to generate `libnnpack.so` and place where `LD_LIBRARY_PATH` can find it.
