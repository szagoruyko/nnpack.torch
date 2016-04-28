require 'nn'
local ffi = require 'ffi'
local nnpack = require 'nnpack.env'
nnpack.C = require 'nnpack.ffi'

local errcheck = function(f, ...)
   local status = nnpack.C[f](...)
   if status ~= nnpack.C.nnp_status_success then
      local str = status
      error('Error in NNPACK: ' .. str .. ' ('..f..')')
   end
end
nnpack.errcheck = errcheck

function nnpack.convert(net, dst)
   net:apply(function(x)
      if torch.typename(x):find'SpatialConvolution' and x.dW == 1 and x.dH == 1 then
         torch.setmetatable(x, (dst == nn and 'nn' or 'nnpack') .. '.SpatialConvolution')
      end
   end)
   return net
end

errcheck('nnp_initialize')

nnpack.threadpool = nnpack.C.pthreadpool_create(torch.getnumthreads())
ffi.gc(nnpack.threadpool, nnpack.C.pthreadpool_destroy)

-- disable for now, unstable
nnpack.threadpool = nil

require 'nnpack.SpatialConvolution'

return nnpack
