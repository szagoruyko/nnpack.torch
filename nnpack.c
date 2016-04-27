#include "nnpack.h"
#include <TH/TH.h>

enum nnp_status nnpack_SpatialConvolution_updateOutput(
          THFloatTensor *input,
          THFloatTensor *output,
          THFloatTensor *weight,
          THFloatTensor *bias,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
  THAssert(THFloatTensor_nDimension(input) == 4);
  size_t batch_size = input->size[0];
  size_t inputWidth = input->size[3];
  size_t inputHeight = input->size[2];
  size_t nInputPlane = weight->size[1];
  size_t nOutputPlane = weight->size[0];
  struct nnp_size input_size = {.height = inputHeight, .width = inputWidth};
  struct nnp_padding pad_size = {.top = padH, .bottom = padH, .left = padW, .right = padW};
  struct nnp_size kernel_size = {.height = kH, .width = kW};
  size_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  size_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  THFloatTensor_resize4d(output, batch_size, nOutputPlane, outputHeight, outputWidth);
  if(batch_size == 1)
  {
    enum nnp_status ret = nnp_convolution_inference(nnp_convolution_algorithm_auto,  
	nnp_convolution_kernel_transform_strategy_reuse,
	nInputPlane,
	nOutputPlane,
	input_size,
	pad_size,
	kernel_size,
	THFloatTensor_data(input),
	THFloatTensor_data(weight),
	THFloatTensor_data(bias),
	THFloatTensor_data(output),
	NULL,
	NULL);
    return ret;
  }
  else {
    enum nnp_status ret = nnp_convolution_output(nnp_convolution_algorithm_auto,
	batch_size,
	nInputPlane,
	nOutputPlane,
	input_size,
	pad_size,
	kernel_size,
	THFloatTensor_data(input),
	THFloatTensor_data(weight),
	THFloatTensor_data(bias),
	THFloatTensor_data(output),
	NULL,
	NULL);
    return ret;
  }
  return nnp_status_success;
}
