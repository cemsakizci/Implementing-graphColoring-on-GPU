// Implementation of the CJP algorithm, proposed by Nguyen Quang Anh Pham and Rui Fan, for parallel graph coloring on GPUs.
// Copyright (C) 2020, Cem Sakızcı <sakizcicem@gmail.com>

// This file is part of Implementing-graphColoring-on-GPU.

// Implementing-graphColoring-on-GPU is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Implementing-graphColoring-on-GPU is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with Implementing-graphColoring-on-GPU.  If not, see <http://www.gnu.org/licenses/>.

#include "vertex.h"
#include "eigen_mtx2csr.h"
#include "find_the_color.h"
#include "graphColoring_GPU.h"
int main(int argc, char **argv) {

	// Reading real data.
	if(argc != 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}
	CSRMat Img(10,10); //!< Input matrix, init. with small size.
	
	if(strstr(argv[1], ".mtx")){
		std::vector<std::string> args; //!< Vector of input arguments.
		// Copy the arguments into the elements of the vector.
		std::copy(argv + 1, argv + argc, std::back_inserter(args));
		if(eigen_mtx2csr(Img, args[0]) != true){
			std::cerr << "ERROR! Can't load SYSMatrix. Exiting..." << std::endl;
			exit(1);
			}
	}
	else {
		printf("Unrecognizable input file format\n");
		exit(0);
	}
	Img.makeCompressed();
	if (Img.innerIndexPtr() == NULL)
		printf("Img.innerIndexPtr() is NULL\n");
	if (Img.outerIndexPtr() == NULL)
		printf("Img.outerIndexPtr() is NULL\n");

	int numberOfVertices = (Img.nonZeros() / sizeof(int));
	// Creating all the vertices.
	Vertex *vertices = createVertices(numberOfVertices);
	int *startingIndexOfRows = Img.innerIndexPtr(), *columnIndices = Img.outerIndexPtr();
	
	// Setting count, neighboursIndices, arraySize attributes
	settingVertexAttributes(startingIndexOfRows , columnIndices , vertices, numberOfVertices);

	Img.resize(10,10); // Free the matrix by shrinking it.
	
	// Just to see the internal attributes of each vertex.
	/*for(int i=0; i<numberOfVertices; i++) {
		Vertex currentVertex = vertices[i];
		printf("vertex[%d]:  %d, randomValue: %d, count: %d, \n", i, currentVertex.vertexIndex, currentVertex.randomValue, currentVertex.count);
		for(int j=0; j<currentVertex.arraySize; j++) {
			printf("neighbour[%d]: %d\n", j, currentVertex.neighboursIndices[j]);
		}
	}*/
	// To measure the time elapsed in the kernel function.
	graphColoring_GPU(numberOfVertices,vertices);

}