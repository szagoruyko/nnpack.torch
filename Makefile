TORCH_TH = $(shell dirname `which th`)
TORCH_INCLUDE = $(TORCH_TH)/../include
TORCH_LIB = $(TORCH_TH)/../lib
NNPACK_INCLUDE = ../NNPACK/include
NNPACK_LIB = ../NNPACK/lib/libnnpack.a

libnnpack.so:
		gcc -I$(TORCH_INCLUDE) \
		  -I$(NNPACK_INCLUDE) \
		  -I$(NNPACK_INCLUDE)/../third-party/pthreadpool/include \
		  -L$(TORCH_LIB) \
		  nnpack.c $(NNPACK_LIB) -fPIC  -std=c11 -shared -O3 -lTH -o $@
clean:
		rm libnnpack.so
