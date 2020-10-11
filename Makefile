CXX = g++
CU = nvcc
CUDA_PATH = usr/local/cuda/9.2

CFLAGS = -std=c++11 -w -O3
CUFLAGS = -lcudart
CU_ARCH = -arch=sm_60

P_PATH = graphColoring
P_BIN = $(P_PATH)/bin
P_BUILD = $(P_PATH)/build

all: mtx2csr_directed.o mtx2csr_undirected.o vertex.o find_the_color.o graphColoring_GPU.o test.o main.o


test.o: $(P_PATH)/test.cu
	@echo "Compiling test..."
	$(CU) $(P_BIN)/vertex.o $(P_BIN)/mtx2csr_directed.o $(P_BIN)/find_the_color.o $(P_PATH)/test.cu -o $(P_BUILD)/TEST_directed.o
	$(CU) $(P_BIN)/vertex.o $(P_BIN)/mtx2csr_undirected.o $(P_BIN)/find_the_color.o $(P_PATH)/test.cu -o $(P_BUILD)/TEST_undirected.o
	@echo "Done!"

main.o: $(P_PATH)/main.cpp mtx2csr_directed.o mtx2csr_undirected.o vertex.o find_the_color.o graphColoring_GPU.o 
	@echo "Compiling mainfile..."
	$(CXX) $(P_PATH)/main.cpp -o $(P_BUILD)/main.o $(CFLAGS) $(CUFLAGS) $(P_BIN)/vertex.o $(P_BIN)/graphColoring_GPU.o $(P_BIN)/find_the_color.o -L $(CUDA_PATH)/lib64 -I $(CUDA_PATH)/include
	@echo "Done!"

graphColoring_GPU.o: $(P_PATH)/find_the_color.cu
	@echo "Compiling graphColoring_GPU.cu"
	$(CU) -c $(CU_ARCH) $(P_PATH)/graphColoring_GPU.cu -o $(P_BIN)/graphColoring_GPU.o -L $(CUDA_PATH)/lib64 -I $(CUDA_PATH)/include
	@echo "Done!"

find_the_color.o: $(P_PATH)/find_the_color.cu
	@echo "Compiling find_the_color.cu"
	$(CU) -c $(CU_ARCH) $(P_PATH)/find_the_color.cu -o $(P_BIN)/find_the_color.o -L $(CUDA_PATH)/lib64 -I $(CUDA_PATH)/include
	@echo "Done!"

mtx2csr_directed.o: $(P_PATH)/mtx2csr_directed.cpp
	@echo "Compiling mtx2csr_directed..."
	$(CXX) -c $(P_PATH)/mtx2csr_directed.cpp $(CFLAGS) -o $(P_BIN)/mtx2csr_directed.o
	@echo "Done!"

mtx2csr_undirected.o: $(P_PATH)/mtx2csr_undirected.cpp
	@echo "Compiling mtx2csr_undirected..."
	$(CXX) -c $(P_PATH)/mtx2csr_undirected.cpp $(CFLAGS) -o $(P_BIN)/mtx2csr_undirected.o 
	@echo "Done!"

vertex.o: $(P_PATH)/vertex.cpp
	@echo "Compiling mtx2csr_directed..."
	$(CXX) -c $(P_PATH)/vertex.cpp -o $(P_BIN)/vertex.o $(CFLAGS)
	@echo "Done!"

clean:
	@echo "Cleaning..."
	rm $(P_BIN)/*.o
	rm $(P_BUILD)/*.o
	@echo "All output files removed."