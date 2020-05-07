#include "implementation.h"
/*
---COMPILATION COMMANDS----
$ nvcc vertex.cpp implementation.cu -c
$ nvcc vertex.o implementation.o test.cu -o TEST
$ ./TEST
*/

int main(int argc, char **argv) {

	// manually entered samples
	//int startingIndexOfRows[5] = {0,2,4,6,8};
	//int columnIndices[8] = {1,3,0,2,1,3,0,2};
	int startingIndexOfRows[9] = {0, 3, 6, 9, 12, 14, 16, 19, 22};
	int columnIndices[22] = {1, 2, 7, 0, 3, 6, 0, 4, 5, 1, 6, 7, 2, 7, 2, 6, 1, 3, 5, 0, 3, 4};

	int numberOfVertices = (sizeof(startingIndexOfRows) / sizeof(int)) - 1;

	// Creating all the vertices.
	Vertex *vertices = createVertices(numberOfVertices);

	// Setting count, neighboursIndices, arraySize attributes
	settingVertexAttributes(startingIndexOfRows, columnIndices, vertices, numberOfVertices);

	// Just to see the internal attributes of each vertex.
	/*for(int i=0; i<numberOfVertices; i++) {
		Vertex currentVertex = vertices[i];
		printf("vertex[%d]:  %d, randomValue: %d, count: %d, \n", i, currentVertex.vertexIndex, currentVertex.randomValue, currentVertex.count);
		for(int j=0; j<currentVertex.arraySize; j++) {
			printf("neighbour[%d]: %d\n", j, currentVertex.neighboursIndices[j]);
		}
	}*/
	
	cudaError_t error;

	for(int i=0; i<numberOfVertices; i++) {
		
		// Select the i.th Vertex to be colored.
		Vertex vertexToBeColored = vertices[i];

		// current vertex's neighbor size in bytes
		int neighborSize = sizeof(Vertex) * vertexToBeColored.arraySize;
		
		// memory allocation for vertex's neigbors in the HOST(INPUT).
		Vertex *h_neighbors = (Vertex *) malloc(neighborSize);

		// memory allocation for the color in the HOST(OUTPUT).
		int *h_color = (int *) malloc(sizeof(int));
		
		// Checking if there were any failures while allocating host data.
		if(h_neighbors == NULL || h_color == NULL) {
			fprintf(stderr, "Failed to allocate host data!\n");
			exit(EXIT_FAILURE);
		}

		// filling each neighbor from the vertices array.
		for(int j=0; j<vertexToBeColored.arraySize; j++) {
			h_neighbors[j] = vertices[vertexToBeColored.neighboursIndices[j]]; 
		}

		// allocation of neighbors in the DEVICE(INPUT).
		Vertex *d_neighbors = NULL;
		error = cudaMalloc((void **) &d_neighbors, neighborSize);
		

		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to allocate d_neighbors (error code %s)!\n", cudaGetErrorString(error));
		    exit(EXIT_FAILURE);
		}

		// allocation of the color in the DEVICE(OUTPUT).
		int *d_color = NULL;
		error = cudaMalloc((void **) &d_color, sizeof(int));

		if(error != cudaSuccess) {
		fprintf(stderr, "Failed to allocate d_color (error code %s)!\n", cudaGetErrorString(error));
		    exit(EXIT_FAILURE);
		}

		//Copying of neighbors from host to device.
		printf("Copy neighbors from the host memory to the CUDA device\n");
		error = cudaMemcpy(d_neighbors, h_neighbors, neighborSize,cudaMemcpyHostToDevice);

		if(error != cudaSuccess) {
		    fprintf(stderr, "Failed to copy h_neighbors to device (error code %s)!\n", cudaGetErrorString(error));
		    exit(EXIT_FAILURE);
		}

		int current_phase = 0;
		// Here, the current phase is incremented by 1 until the current vertex finds the appropriate color.
		while(true) {

			printf("CUDA kernel launch for V[%d] in %d.phase\n", i, current_phase);
			find_the_color<<<1,2>>>(d_neighbors, neighborSize, current_phase, d_color);

			error = cudaGetLastError();

			if(error != cudaSuccess) {
				fprintf(stderr, "Failed to launch the kernel (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Copy d_color to host.
			printf("Copy d_color from the CUDA device to the host memory\n");
			error = cudaMemcpy(h_color, d_color, sizeof(int), cudaMemcpyDeviceToHost);

			if(error != cudaSuccess) {
				fprintf(stderr, "Failed to copy d_color from device to host (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			printf("Copying has successfully completed\n");

			if(*h_color != -1) {
				printf("color of V[%d]= %d\n", i, *h_color);
				vertices[i].color = *h_color;
				break;
			}
			
			printf("Going to the next phase...\n");
			current_phase++;
		}

		// Free device global memory
		error = cudaFree(d_neighbors);

		if(error != cudaSuccess) {
		    fprintf(stderr, "Failed to free d_neighbors (error code %s)!\n", cudaGetErrorString(error));
		    exit(EXIT_FAILURE);
		}

		error = cudaFree(d_color);

		if(error != cudaSuccess) {
		    fprintf(stderr, "Failed to free d_color (error code %s)!\n", cudaGetErrorString(error));
		    exit(EXIT_FAILURE);
		}

		// Free host memory
		free(h_neighbors);
		free(h_color);

	}

	printf("DONE\n");

	return 0;
	
}
