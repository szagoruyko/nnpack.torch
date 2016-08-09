Torch FFI-bindings for NNPACK 
=============

NNPACK is a fast CPU implementation of convolution operations for training ConvNets:
 
https://github.com/Maratyszcza/NNPACK

The bindings are fully working and tested against `nn` version. Only single precision supported.
Make sure you have AVX2 compatible Skylake/Broadwell/Haswell CPU.

Limitations of NNPACK:

 * there is no scale parameter on `accGradParameters` call

# Installation

Follow installation steps at https://github.com/Maratyszcza/NNPACK to generate `libnnpack.so` and place where `LD_LIBRARY_PATH` can find it.

Then do

```
luarocks install https://raw.githubusercontent.com/szagoruyko/nnpack.torch/master/nnpack-scm-1.rockspec
```

# Conversion between nnpack and nn

Similar to `cudnn.convert` in [cudnn.torch](https://github.com/soumith/cudnn.torch) easy backend switching is supported. To switch to `nnpack` just do:

```lua
nnpack.convert(net, nnpack)
```

There will be no memory copy, just metatables will be swapped.

# Credits

Thanks to @Maratyszcza for adding the option to generate shared NNPACK library.
