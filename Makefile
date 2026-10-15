NVCC  ?= nvcc
ARCH  ?= native
FLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo

softmax: softmax.cu
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f softmax results.csv

.PHONY: clean
