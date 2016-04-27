local nnpack = require 'env'
local ffi = require 'ffi'

ffi.cdef[[
/**
 * @brief Status code for any NNPACK function call.
 */
enum nnp_status {
	/** The call succeeded, and all output arguments now contain valid data. */
	nnp_status_success = 0,
	/** NNPACK function was called with batch_size == 0. */
	nnp_status_invalid_batch_size = 2,
	/** NNPACK function was called with channels == 0. */
	nnp_status_invalid_channels = 3,
	/** NNPACK function was called with input_channels == 0. */
	nnp_status_invalid_input_channels = 4,
	/** NNPACK function was called with output_channels == 0. */
	nnp_status_invalid_output_channels = 5,
	/** NNPACK function was called with input_size.height == 0 or input_size.width == 0 */
	nnp_status_invalid_input_size = 10,
	/** NNPACK function was called with input_stride.height == 0 or input_stride.width == 0 */
	nnp_status_invalid_input_stride = 11,
	/** NNPACK function was called with input_padding not less than respective kernel (or pooling) size, i.e.:
	 *
	 *  - input_padding.left   >= kernel_size.width  (>= pooling_size.width)
	 *  - input_padding.right  >= kernel_size.width  (>= pooling_size.width)
	 *  - input_padding.top    >= kernel_size.height (>= pooling_size.height)
	 *  - input_padding.bottom >= kernel_size.height (>= pooling_size.height)
	 */
	nnp_status_invalid_input_padding = 12,
	/** NNPACK function was called with kernel_size.height == 0 or kernel_size.width == 0 */
	nnp_status_invalid_kernel_size = 13,
	/** NNPACK function was called with pooling_size.height == 0 or pooling_size.width == 0 */
	nnp_status_invalid_pooling_size = 14,
	/** NNPACK function was called with pooling_stride.height == 0 or pooling_stride.width == 0 */
	nnp_status_invalid_pooling_stride = 15,
	/** NNPACK function was called with convolution algorithm not in nnp_convolution_algorithm enumeration */
	nnp_status_invalid_algorithm = 15,

	/** NNPACK does not support the particular input size for the function */
	nnp_status_unsupported_input_size = 20,
	/** NNPACK does not support the particular input stride for the function */
	nnp_status_unsupported_input_stride = 21,
	/** NNPACK does not support the particular input padding for the function */
	nnp_status_unsupported_input_padding = 22,
	/** NNPACK does not support the particular kernel size for the function */
	nnp_status_unsupported_kernel_size = 23,
	/** NNPACK does not support the particular pooling size for the function */
	nnp_status_unsupported_pooling_size = 24,
	/** NNPACK does not support the particular pooling stride for the function */
	nnp_status_unsupported_pooling_stride = 25,
	/** NNPACK does not support the particular convolution algorithm for the function */
	nnp_status_unsupported_algorithm = 26,

	/** NNPACK function was called before the library was initialized */
	nnp_status_uninitialized = 50,
	/** NNPACK does not implement this function for the host CPU */
	nnp_status_unsupported_hardware = 51,
	/** NNPACK failed to allocate memory for temporary buffers */
	nnp_status_out_of_memory = 52
};

enum nnp_status nnp_initialize(void);

enum nnp_status nnp_deinitialize(void);

enum nnp_status nnpack_SpatialConvolution_updateOutput(
//void nnpack_SpatialConvolution_updateOutput(
          THFloatTensor *input,
          THFloatTensor *output,
          THFloatTensor *weight,
          THFloatTensor *bias,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH);
]]

local C = ffi.load'./libnnpack.so'
return C
