#include <string.h>
#include "mtx2csr.h"
#include "implementation.h"
#include "common.h"
/*
---COMPILATION COMMANDS----
$ nvcc vertex.cpp mtx2csr.cpp implementation.cu -c
$ nvcc vertex.o mtx2csr.o implementation.o test.cu -o TEST
$ ./TEST <graph-name>
*/

int main(int argc, char **argv) {

	// manually entered samples
	//int startingIndexOfRows[5] = {0,2,4,6,8};
	//int columnIndices[8] = {1,3,0,2,1,3,0,2};

	// Reading real data.
	if(argc != 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}

	int numberOfVertices, nnz, *startingIndexOfRows = NULL, *columnIndices = NULL;
	if(strstr(argv[1], ".mtx"))
		mtx2csr(argv[1], numberOfVertices, nnz, startingIndexOfRows, columnIndices);
	else {
		printf("Unrecognizable input file format\n");
		exit(0);
	}

	if (startingIndexOfRows == NULL)
		printf("startingIndexOfRows is NULL\n");
	if (columnIndices == NULL)
		printf("columnIndices is NULL\n");

	//int startingIndexOfRows[9] = {0, 3, 6, 9, 12, 14, 16, 19, 22};
	//int columnIndices[22] = {1, 2, 7, 0, 3, 6, 0, 4, 5, 1, 6, 7, 2, 7, 2, 6, 1, 3, 5, 0, 3, 4};

	//int numberOfVertices = (sizeof(startingIndexOfRows) / sizeof(int)) - 1;

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
	/*
	// For cuda events.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	*/

	// To measure time elapsed in the kernel function.
	double t1, t2, total_elapsed_time = 0.0;
	//float total_elapsed_time = 0.0f;
	int max_color = 0;

	while(true) {

		cudaError_t error;

		int numberOfVerticesToBeColored = 0;
		// In this for loop, the number of vertices, having count value as 0, is determined.
		for(int i=0; i<numberOfVertices; i++) {
			if(vertices[i].count == 0 && vertices[i].color == -1) 
				numberOfVerticesToBeColored++;
		}
		printf("# VERTICES TO BE COLORED : %d\n", numberOfVerticesToBeColored);

		// Checking if all vertices have been colored.
		if(numberOfVerticesToBeColored == 0) {
			break;
		}

		// Allocation on host memory for vertices that will be colored.
		Vertex *verticesToBeColored = (Vertex *) malloc(sizeof(Vertex) * numberOfVerticesToBeColored);

		// Allocation on host memory for neighbor sizes of vertices(INPUT).
		int *h_neighborSizeArray = (int *) malloc(sizeof(int) * numberOfVerticesToBeColored);
		
		// Memory allocation on host for the colors(OUTPUT).
		int *h_colors_found = (int *) malloc(sizeof(int) * numberOfVerticesToBeColored);

		// Checking if there were any failures while allocating host data.
		if(h_neighborSizeArray == NULL || h_colors_found == NULL) {
			fprintf(stderr, "Failed to allocate host data!\n");
			exit(EXIT_FAILURE);
		}

		// Initializing h_colors_found with -1.
		for(int i=0; i<numberOfVerticesToBeColored; i++) {
			h_colors_found[i] = -1;
		}

		// Setting verticesToBeColored and h_neighborSizeArray.
		int totalNumberOfNeighbors = 0, individualNeighborNumber = 0;
		for(int i=0, j=0; i<numberOfVertices; i++) {
			if(vertices[i].count == 0 && vertices[i].color == -1) {
				verticesToBeColored[j] = vertices[i];
				individualNeighborNumber = vertices[i].arraySize;
				h_neighborSizeArray[j] = individualNeighborNumber;
				totalNumberOfNeighbors += individualNeighborNumber; //Accumulating the neighbor size of each vertex to obtain the total for memory allocation. 
				j++;
			}
		}

		// Memory allocation on host for all neighbors of all vertices(INPUT).
		int neighborsSizeInBytes = sizeof(Vertex) * totalNumberOfNeighbors;
		Vertex *h_neighborsOfAllVertices = (Vertex *) malloc(neighborsSizeInBytes);

		// Checking if there were any failures while allocating host data.
		if(h_neighborsOfAllVertices == NULL) {
			fprintf(stderr, "Failed to allocate host data!\n");
			exit(EXIT_FAILURE);
		}

		// Setting h_neighborsOfAllVertices. 
		int cumulativeIndex = 0;
		for(int i=0; i<numberOfVerticesToBeColored; i++) {
			int neighborSize = h_neighborSizeArray[i];
			for(int j=0; j<neighborSize; j++) {
				h_neighborsOfAllVertices[cumulativeIndex] = vertices[verticesToBeColored[i].neighboursIndices[j]];
				cumulativeIndex++;
			}
		}

		// Device allocation of all neighbors(INPUT).
		Vertex *d_neighborsOfAllVertices = NULL;
		error = cudaMalloc((void **) &d_neighborsOfAllVertices, neighborsSizeInBytes);
			
		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to allocate d_neighborsOfAllVertices (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		
		// Device allocation of neighbor sizes(INPUT).
		int *d_neighborSizeArray = NULL;
		error = cudaMalloc((void **) &d_neighborSizeArray, numberOfVerticesToBeColored * sizeof(int));
			
		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to allocate d_neighborSizeArray (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		// Device allocation of colors_found(OUTPUT).
		int *d_colors_found = NULL;
		error = cudaMalloc((void **) &d_colors_found, numberOfVerticesToBeColored * sizeof(int));
			
		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to allocate d_colors_found (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		
		// Copying of colors from host to device.
		printf("Copy colors from the host memory to the CUDA device\n");
		error = cudaMemcpy(d_colors_found, h_colors_found, numberOfVerticesToBeColored * sizeof(int), cudaMemcpyHostToDevice);

		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to copy h_colors_found to device (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		//Copying of all neighbors from host to device.
		printf("Copy all neighbors from the host memory to the CUDA device\n");
		error = cudaMemcpy(d_neighborsOfAllVertices, h_neighborsOfAllVertices, neighborsSizeInBytes, cudaMemcpyHostToDevice);

		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to copy h_neighborsOfAllVertices to device (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		//Copying of neighborSizeArray from host to device.
		printf("Copy neighborSizeArray from the host memory to the CUDA device\n");
		error = cudaMemcpy(d_neighborSizeArray, h_neighborSizeArray, numberOfVerticesToBeColored * sizeof(int), cudaMemcpyHostToDevice);

		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to copy h_neighborSizeArray to device (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		int current_phase = 0;
		int threads_per_block = 32;
		// Here, the current phase is incremented by 1 until all vertices to be colored find their appropriate colors.
		while(true) {

			printf("CUDA kernel launch in %d.phase\n", current_phase);
			t1 = rtclock();
			//cudaEventRecord(start);
			find_the_color<<<numberOfVerticesToBeColored, threads_per_block>>>(d_neighborsOfAllVertices, d_neighborSizeArray, current_phase, d_colors_found);
			t2 = rtclock();
			//cudaEventRecord(stop);
			/*
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			total_elapsed_time += milliseconds;
			*/
			total_elapsed_time += 1000.0f * (t2 - t1);

			error = cudaGetLastError();

			if(error != cudaSuccess) {
				fprintf(stderr, "Failed to launch the kernel (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Copy d_colors_found to host.
			printf("Copy d_colors_found from the CUDA device to the host memory\n");
			error = cudaMemcpy(h_colors_found, d_colors_found, sizeof(int) * numberOfVerticesToBeColored, cudaMemcpyDeviceToHost);

			if(error != cudaSuccess) {
				fprintf(stderr, "Failed to copy d_colors_found from device to host (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			printf("Copying has successfully completed\n");

			// Finding if any vertex from verticesToBeColored array could not be colored for the current phase.
			int numberOfUncoloredVertices = 0;
			for(int i=0; i<numberOfVerticesToBeColored; i++) {
				if(h_colors_found[i] != -1) {
					if(h_colors_found[i] >= 256*current_phase+1 && h_colors_found[i] <= 256*(current_phase+1)) {
						int index = verticesToBeColored[i].vertexIndex;
						int color = h_colors_found[i];
						printf("color of V[%d] = %d\n", index, color);
						vertices[index].color = color; // setting the color.
						// To find the maximum color used.
						if(color > max_color) {
							max_color = color;
						}

						// Decrement neighbor's count values by 1.
						int neighborSize = vertices[index].arraySize;
						for(int j=0; j<neighborSize; j++) {
							printf("N_id: %d, N_color: %d\n",vertices[index].neighboursIndices[j], vertices[vertices[index].neighboursIndices[j]].color);
							if(vertices[vertices[index].neighboursIndices[j]].count > 0)
								vertices[vertices[index].neighboursIndices[j]].count--;
						}
					}
					
				}
				else {
					numberOfUncoloredVertices++;
				}
			}	
			
			if(numberOfUncoloredVertices > 0) {
				
				printf("Going to the next phase...\n");
				current_phase++;
				
			}
			else {
				break;
			}
			
		}
		// Free device global memory
		error = cudaFree(d_neighborsOfAllVertices);

		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to free d_neighborsOfAllVertices (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		error = cudaFree(d_neighborSizeArray);

		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to free d_neighborSizeArray (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		error = cudaFree(d_colors_found);

		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to free d_colors_found (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		printf("DELETED ALLOCATED MEMORY FOR DEVICE\n");

		// Free host memory
		free(h_neighborsOfAllVertices);
		free(h_neighborSizeArray);
		free(h_colors_found);
		printf("DELETED ALLOCATED MEMORY FOR HOST \n");

	}
	/*
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	*/
	printf("-----ELAPSED TIME ---> %f ms\n", total_elapsed_time);
	printf("max color value: %d\n", max_color);

	printf("DONE\n");

	return 0;
	
}
