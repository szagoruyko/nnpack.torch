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
   -- backward compatibility
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   end
   input = makeContiguous(self, input)
   nnpack.errcheck('nnpack_SpatialConvolution_updateOutput',
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.kW, self.kH,
      self.padW, self.padH
   )
   return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
   if not self.gradInput then return end
   input, gradOutput = makeContiguous(self, input, gradOutput)
   nnpack.errcheck('nnpack_SpatialConvolution_updateGradInput',
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.kW, self.kH,
      self.padW, self.padH
   )
   return self.gradInput
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   assert(scale == 1)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   assert((self.bias and self.gradBias) or (self.bias == nil and self.gradBias == nil))
   nnpack.errcheck('nnpack_SpatialConvolution_accGradParameters',
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      self.gradBias:cdata(),
      self.kW, self.kH,
      self.padW, self.padH)
end

