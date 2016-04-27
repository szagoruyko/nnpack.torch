package = "nnpack"
version = "scm-1"

source = {
   url = "git://github.com/szagoruyko/nnpack.torch",
}

description = {
   summary = "Extra Lua functions.",
   detailed = [[
NNPACK torch nn port
   ]],
   homepage = "https://github.com/szagoruyko/nnpack.torch",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "image",
}

build = {
   type = "builtin",
   modules = {
      ['nnpack.init'] = 'init.lua',
      ['nnpack.env'] = 'env.lua',
      ['nnpack.ffi'] = 'ffi.lua',
      ['nnpack.SpatialConvolution'] = 'SpatialConvolution.lua',
   }
}
