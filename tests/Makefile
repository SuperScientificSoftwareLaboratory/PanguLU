MPICC ?= mpicc
PRECISION ?= R64
CFLAGS ?= -O3 -g -fno-omit-frame-pointer
CPPFLAGS += -DCALCULATE_TYPE_$(PRECISION) -I../include
BLAS_LIBS ?= -lopenblas
CUDA_LIBS ?=
PANGULU_LIB ?= ../lib/libpangulu.a
OUTPUT ?= ../.build/tests
TARGETS = $(OUTPUT)/numerical-$(PRECISION) $(OUTPUT)/errors-$(PRECISION)

.PHONY: all check
all: $(TARGETS)
$(OUTPUT):
	mkdir -p $@
$(OUTPUT)/numerical-$(PRECISION): numerical_contracts.c test_scalar.h $(PANGULU_LIB) ../include/pangulu.h ../include/pangulu_interface_common.h | $(OUTPUT)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) $< $(PANGULU_LIB) $(CUDA_LIBS) $(BLAS_LIBS) -fopenmp -lpthread -lm -o $@
$(OUTPUT)/errors-$(PRECISION): error_contracts.c test_scalar.h $(PANGULU_LIB) ../include/pangulu.h ../include/pangulu_interface_common.h | $(OUTPUT)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) $< $(PANGULU_LIB) $(CUDA_LIBS) $(BLAS_LIBS) -fopenmp -lpthread -lm -o $@
check: all
	$(OUTPUT)/numerical-$(PRECISION)
	$(OUTPUT)/errors-$(PRECISION)
