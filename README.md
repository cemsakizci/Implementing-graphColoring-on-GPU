# Implementing-graphColoring-on-GPU

The purpose is to implement the CJP algorithm on the GPU. This algorithm has been proposed in the paper, "Efficient Algorithms for Graph Coloring on GPU", by Quang Anh Pham Nguyen and Rui Fan.

NOTE: The paper is available in the repository. 

The implementation is specific for NVIDIA GPUs because of using CUDA.

I have used the "mtx2csr" function from the following link to convert ".mtx" files into a proper csr format https://github.com/chenxuhao/csrcolor/blob/master/src/csrcolor/csrcolor.cu
Also, I have modified it according to the current project needs, so you can check the details.

I have used the "rtclock" function to measure the execution time of graph coloring from the following link : https://github.com/chenxuhao/csrcolor/blob/master/include/common.h
