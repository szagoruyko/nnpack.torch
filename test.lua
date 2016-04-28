local nnpack = require 'nnpack'

local nnpacktest = torch.TestSuite()
local precision_forward = 1e-4
local precision_backward = 1e-2
local precision_jac = 1e-3
local precision_io = 1e-5
local nloop = 1
local times = {}
local mytester
local jac = nn.Jacobian


function nnpacktest.SpatialConvolution_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si,sj = 1,1
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(1,from,inj,ini):float()
   local sconv = nn.SpatialConvolution(from,to,ki,kj,si,sj):float()
   local gconv = nnpack.SpatialConvolution(from,to,ki,kj,si,sj):float()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)

   local function test(sconv, gconv)
     local groundtruth = sconv:forward(input)
     local resfloat = gconv:forward(input)
     mytester:asserteq(resfloat:dim(), 4, 'error in dimension')
     local error = resfloat:float() - groundtruth:float()
     mytester:assertlt(error:abs():max(), precision_forward,
                       'error on state (forward) ')

     -- IO
     local ferr,berr = jac.testIO(gconv, input)
     mytester:assertlt(ferr, precision_io, torch.typename(gconv) .. ' - i/o forward err ')
     mytester:assertlt(berr, precision_io, torch.typename(gconv) .. ' - i/o backward err ')
   end

   test(sconv, gconv)

   local gconv = nnpack.convert(sconv, nnpack)
   mytester:asserteq(torch.typename(gconv), 'nnpack.SpatialConvolution', 'conversion type check')
   test(sconv, gconv)
end

function nnpacktest.SpatialConvolution_forward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   -- local si = math.random(1,ki)
   -- local sj = math.random(1,kj)
   local si,sj = 1,1
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(bs,from,inj,ini):float()
   local sconv = nn.SpatialConvolution(from,to,ki,kj,si,sj):float()
   local gconv = nnpack.SpatialConvolution(from,to,ki,kj,si,sj):float()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)

   local function test(sconv, gconv)
     local groundtruth = sconv:forward(input)
     local rescuda = gconv:forward(input)
     local error = rescuda:float() - groundtruth:float()
     mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')

     -- IO
     local ferr,berr = jac.testIO(gconv, input)
     mytester:assertlt(ferr, precision_io, torch.typename(gconv) .. ' - i/o forward err ')
     mytester:assertlt(berr, precision_io, torch.typename(gconv) .. ' - i/o backward err ')
   end

   test(sconv, gconv)

   local gconv = nnpack.convert(sconv, nnpack)
   mytester:asserteq(torch.typename(gconv), 'nnpack.SpatialConvolution', 'conversion type check')
   test(sconv, gconv)
end

function nnpacktest.SpatialConvolution_backward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   -- local si = math.random(1,ki)
   -- local sj = math.random(1,kj)
   local si,sj = 1,1
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local scale = 1

   local input = torch.randn(bs,from,inj,ini):float()
   local gradOutput = torch.randn(bs,to,outj,outi):float()
   local sconv = nn.SpatialConvolution(from,to,ki,kj,si,sj):float()
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput, scale)
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = nnpack.SpatialConvolution(from,to,ki,kj,si,sj):float()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   gconv:forward(input)

   local function test(sconv, gconv)
     gconv:forward(input)
     gconv:zeroGradParameters()
     local rescuda = gconv:backward(input, gradOutput, scale)
     local weightcuda = gconv.gradWeight
     local biascuda = gconv.gradBias

     local error = rescuda - groundgrad
     local werror = weightcuda - groundweight
     local berror = biascuda - groundbias

     mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
     mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
     mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
   end

   test(sconv, gconv)
   local gconv = nnpack.convert(sconv, cudnn)
   mytester:asserteq(torch.typename(gconv), 'nnpack.SpatialConvolution', 'conversion type check')
   test(sconv, gconv)
end


torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
mytester = torch.Tester()
mytester:add(nnpacktest)

mytester:run()
