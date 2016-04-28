local nnpack = require 'nnpack.env'
local SpatialConvolution, parent = torch.class('nnpack.SpatialConvolution', 'nn.SpatialConvolution', nnpack)

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
	 self._gradOutput = self._gradOutput or gradOutput.new()
	 self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
	 gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function SpatialConvolution:updateOutput(input)
   assert(self.dW == 1 and self.dH == 1)
   assert(input:dim() == 4)
   -- backward compatibility
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   end
   input = makeContiguous(self, input)

   local batch_size = input:size(1)
   local inputWidth, inputHeight = input:size(4), input:size(3)
   local nOutputPlane, nInputPlane = self.weight:size(1), self.weight:size(2)
   local outputWidth = inputWidth + 2*self.padW - self.kW + 1
   local outputHeight = inputHeight + 2*self.padH - self.kH + 1
   local input_size = {width = inputWidth, height = inputHeight}
   local pad_size = {top = self.padH, bottom = self.padH,
         left = self.padW, right = self.padH}
   local kernel_size = {width = self.kW, height = self.kH}

   self.output:resize(batch_size, nOutputPlane, outputHeight, outputWidth)

   if batch_size == 1 then
      nnpack.errcheck('nnp_convolution_inference',
         nnpack.C.nnp_convolution_algorithm_auto,
         nnpack.C.nnp_convolution_kernel_transform_strategy_reuse,
         nInputPlane, nOutputPlane,
         input_size, pad_size, kernel_size,
         input:data(),
         self.weight:data(),
         self.bias:data(),
         self.output:data(),
         nil,
         nil)
   else
      nnpack.errcheck('nnp_convolution_output',
         nnpack.C.nnp_convolution_algorithm_auto,
         batch_size,
         nInputPlane, nOutputPlane,
         input_size, pad_size, kernel_size,
         input:data(),
         self.weight:data(),
         self.bias:data(),
         self.output:data(),
         nil,
         nil)
   end
   return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
   if not self.gradInput then return end
   input, gradOutput = makeContiguous(self, input, gradOutput)
   self.gradInput:resizeAs(input)

   local batch_size = input:size(1)
   local inputWidth, inputHeight = input:size(4), input:size(3)
   local nOutputPlane, nInputPlane = self.weight:size(1), self.weight:size(2)
   local input_size = {width = inputWidth, height = inputHeight}
   local pad_size = {top = self.padH, bottom = self.padH,
         left = self.padW, right = self.padH}
   local kernel_size = {width = self.kW, height = self.kH}

   nnpack.errcheck('nnp_convolution_input_gradient',
      nnpack.C.nnp_convolution_algorithm_auto,
      batch_size,
      nInputPlane, nOutputPlane,
      input_size, pad_size, kernel_size,
      gradOutput:data(),
      self.weight:data(),
      self.gradInput:data(),
      nil,
      nil)
   return self.gradInput
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   assert(scale == 1)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   assert((self.bias and self.gradBias) or (self.bias == nil and self.gradBias == nil))

   local batch_size = input:size(1)
   local inputWidth, inputHeight = input:size(4), input:size(3)
   local nOutputPlane, nInputPlane = self.weight:size(1), self.weight:size(2)
   local input_size = {width = inputWidth, height = inputHeight}
   local pad_size = {top = self.padH, bottom = self.padH,
         left = self.padW, right = self.padH}
   local kernel_size = {width = self.kW, height = self.kH}

   nnpack.errcheck('nnp_convolution_kernel_gradient',
      nnpack.C.nnp_convolution_algorithm_auto,
      batch_size,
      nInputPlane, nOutputPlane,
      input_size, pad_size, kernel_size,
      input:data(),
      gradOutput:data(),
      self.gradWeight:data(),
      nil,
      nil)
end

