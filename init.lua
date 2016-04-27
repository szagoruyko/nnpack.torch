require 'nn'
local nnpack = require 'nnpack.env'
nnpack.C = require 'nnpack.ffi'
nnpack.C.nnp_initialize()

local errcheck = function(f, ...)
   local status = nnpack.C[f](...)
   if status ~= nnpack.C.nnp_status_success then
      local str = status
      error('Error in NNPACK: ' .. str .. ' ('..f..')')
   end
end
nnpack.errcheck = errcheck

require 'nnpack.SpatialConvolution'

function nnpack.convert(net, dst)
   net:apply(function(x)
      if torch.typename(x):find'SpatialConvolution' and x.dW == 1 and x.dH == 1 then
         torch.setmetatable(x, (dst == nn and 'nn' or 'nnpack') .. '.SpatialConvolution')
      end
   end)
   return net
end

return nnpack
