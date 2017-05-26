local nnpack = require 'nnpack.env'

function nnpack.conv2d(input, weight, dW, dH, padW, padH)
   nnpack.typecheck(input, weight)
   dW, dH = dW or 1, dH or 1
   padW, padH = padW or 0, padH or 0
   input = input:contiguous()
   weight = weight:contiguous()

   local batch_size = input:size(1)
   local nOutputPlane, nInputPlane, kH, kW = table.unpack(weight:size():totable())
   local inputWidth, inputHeight = input:size(4), input:size(3)
   local outputWidth  = math.floor((inputWidth + 2*padW - kW) / dW) + 1
   local outputHeight = math.floor((inputHeight + 2*padH - kH) / dH) + 1
   local input_size = {width = inputWidth, height = inputHeight}
   local pad_size = {top = padH, bottom = padH, left = padW, right = padW}
   local kernel_size = {width = kW, height = kH}
   local stride = {width = dW, height = dH}

   local output = input.new(batch_size, nOutputPlane, outputHeight, outputWidth)

   -- no bias is not supported
   local bias = weight.new(nOutputPlane):zero()

   if batch_size == 1 then
      nnpack.errcheck('nnp_convolution_inference',
         nnpack.C.nnp_convolution_algorithm_auto,
         nnpack.C.nnp_convolution_transform_strategy_tuple_based,
         nInputPlane, nOutputPlane,
         input_size, pad_size, kernel_size, stride,
         input:data(),
         weight:data(),
         bias:data(),
         output:data(),
         nnpack.threadpool,
         nil)
   else
      nnpack.errcheck('nnp_convolution_output',
         nnpack.C.nnp_convolution_algorithm_auto,
         batch_size,
         nInputPlane, nOutputPlane,
         input_size, pad_size, kernel_size,
         input:data(),
         weight:data(),
         bias:data(),
         output:data(),
         nnpack.threadpool,
         nil)
   end
   return output
end


function nnpack.conv2d_updateGradInput(input, weight, gradOutput, dW, dH, padW, padH)
   nnpack.typecheck(input, weight, gradOutput)
   weight, gradOutput = weight:contiguous(), gradOutput:contiguous()
   dW, dH = dW or 1, dH or 1
   padW, padH = padW or 0, padH or 0
   local gradInput = input.new(#input)
   nnpack.errcheck('nnp_convolution_input_gradient',
      nnpack.C.nnp_convolution_algorithm_auto,
      input:size(1),
      weight:size(2), weight:size(1),
      {width = input:size(4), height = input:size(3)},
      {top = padH, bottom = padH, left = padW, right = padW},
      {width = weight:size(4), height = weight:size(3)},
      gradOutput:data(),
      weight:data(),
      gradInput:data(),
      nnpack.threadpool,
      nil)
   return gradInput
end


function nnpack.conv2d_accGradParameters(input, weight, gradOutput, dW, dH, padW, padH)
   nnpack.typecheck(input, weight, gradOutput)
   input, gradOutput = input:contiguous(), gradOutput:contiguous()
   local gradWeight = weight.new(#weight)
   nnpack.errcheck('nnp_convolution_kernel_gradient',
      nnpack.C.nnp_convolution_algorithm_auto,
      input:size(1),
      weight:size(2), weight:size(1),
      {width = input:size(4), height = input:size(3)},
      {top = padH, bottom = padH, left = padW, right = padW},
      {width = weight:size(4), height = weight:size(3)},
      input:data(),
      gradOutput:data(),
      gradWeight:data(),
      nnpack.threadpool,
      nil)
   return gradWeight
end

