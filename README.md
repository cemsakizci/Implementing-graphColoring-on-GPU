# Implementing-graphColoring-on-GPU

The purpose is to implement the CJP algorithm on the GPU. This algorithm has been proposed in the paper, "Efficient Algorithms for Graph Coloring on GPU", by Quang Anh Pham Nguyen and Rui Fan.

NOTE: The paper is available in the repository. 

The implementation is specific for NVIDIA GPUs because we use CUDA.

I have also utilized from the csrcolor function, implemented in the cuSPARSE library of NVIDIA, to check its execution since it is a GPU implementation.

Inside "csrcolor.cu" there is a function, called "mtx2csr", to convert ".mtx" files into a proper csr format. I have modified it according to the current project needs, but you can find the original one from the following link : https://github.com/chenxuhao/csrcolor/blob/master/src/csrcolor/csrcolor.cu
