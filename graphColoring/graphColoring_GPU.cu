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


#include "graphColoring_GPU.cuh" 


extern void graphColoring_GPU(int numberOfVertices, Vertex *vertices){
		
    cudaEvent_t start,  //!< CUDA Timer start.
        		stop;   //!< CUDA Timer stop.
    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float total_elapsed_time, milliseconds = 0;
	int max_color = 0;
	while(true) {
		cudaError_t error;
		int numberOfVerticesToBeColored(0);  //!< Number of the vertices that should be colored.
		// In this for loop, the number of vertices, having count value as 0, is determined.
		for(int i=0; i<numberOfVertices; i++) {
			if(vertices[i].count == 0 && vertices[i].color == -1) 
				numberOfVerticesToBeColored++;
		}
		//printf("# VERTICES TO BE COLORED : %d\n", numberOfVerticesToBeColored); // UNCOMMENT if you need to see this additional information.

		// Checking if all vertices have been colored.
		if(numberOfVerticesToBeColored == 0) {
			break;
		}

		Vertex *verticesToBeColored; //!< Vertices to be colored in vertex form.

		int *h_neighborSizeArray,    //!< Array that contains size of the neighbors.
			*h_colors_found;    	 //!< Array that contains the colors found.

		// Allocation on host memory for vertices that will be colored.
		verticesToBeColored = (Vertex *) malloc(sizeof(Vertex) * numberOfVerticesToBeColored);
		// Allocation on host memory for neighbor sizes of vertices(INPUT).
		h_neighborSizeArray  = (int *) malloc(sizeof(int) * numberOfVerticesToBeColored);
		// Memory allocation on host for the colors(OUTPUT).
		h_colors_found = (int *) malloc(sizeof(int) * numberOfVerticesToBeColored);

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
		//printf("Copy colors from the host memory to the CUDA device\n"); // UNCOMMENT if you need to see this additional information.
		error = cudaMemcpy(d_colors_found, h_colors_found, numberOfVerticesToBeColored * sizeof(int), cudaMemcpyHostToDevice);

		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to copy h_colors_found to device (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		//Copying of all neighbors from host to device.
		//printf("Copy all neighbors from the host memory to the CUDA device\n"); // UNCOMMENT if you need to see this additional information.
		error = cudaMemcpy(d_neighborsOfAllVertices, h_neighborsOfAllVertices, neighborsSizeInBytes, cudaMemcpyHostToDevice);

		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to copy h_neighborsOfAllVertices to device (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		//Copying of neighborSizeArray from host to device.
		//printf("Copy neighborSizeArray from the host memory to the CUDA device\n"); // UNCOMMENT if you need to see this additional information.
		error = cudaMemcpy(d_neighborSizeArray, h_neighborSizeArray, numberOfVerticesToBeColored * sizeof(int), cudaMemcpyHostToDevice);

		if(error != cudaSuccess) {
			fprintf(stderr, "Failed to copy h_neighborSizeArray to device (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		int current_phase = 0;
		int threads_per_block = 32;
		// Here, the current phase is incremented by 1 until all vertices to be colored find their appropriate colors.
		while(true) {

			//printf("CUDA kernel launch in %d.phase\n", current_phase); // UNCOMMENT if you need to see this additional information.
			cudaEventRecord(start);
			find_the_color<<<numberOfVerticesToBeColored, threads_per_block>>>(d_neighborsOfAllVertices, d_neighborSizeArray, current_phase, d_colors_found);
			cudaEventRecord(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			total_elapsed_time += 1000.0f * (milliseconds);

			error = cudaGetLastError();

			if(error != cudaSuccess) {
				fprintf(stderr, "Failed to launch the kernel (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Copy d_colors_found to host.
			//printf("Copy d_colors_found from the CUDA device to the host memory\n"); // UNCOMMENT if you need to see this additional information.
			error = cudaMemcpy(h_colors_found, d_colors_found, sizeof(int) * numberOfVerticesToBeColored, cudaMemcpyDeviceToHost);

			if(error != cudaSuccess) {
				fprintf(stderr, "Failed to copy d_colors_found from device to host (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			//printf("Copying has successfully completed\n"); // UNCOMMENT if you need to see this additional information.

			// Finding if any vertex from verticesToBeColored array could not be colored for the current phase.
			int numberOfUncoloredVertices = 0;
			for(int i=0; i<numberOfVerticesToBeColored; i++) {
				if(h_colors_found[i] != -1) {
					if(h_colors_found[i] >= 256*current_phase+1 && h_colors_found[i] <= 256*(current_phase+1)) {
						int index = verticesToBeColored[i].vertexIndex;
						int color = h_colors_found[i];
						//printf("color of V[%d] = %d\n", index, color); // UNCOMMENT if you need to see this additional information.
						vertices[index].color = color; // setting the color.
						// To find the maximum color used.
						if(color > max_color) {
							max_color = color;
						}

						// Decrement neighbor's count values by 1.
						int neighborSize = vertices[index].arraySize;
						for(int j=0; j<neighborSize; j++) {
							//printf("N_id: %d, N_color: %d\n",vertices[index].neighboursIndices[j], vertices[vertices[index].neighboursIndices[j]].color); // UNCOMMENT if you need to see this additional information.
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

		//printf("DELETED ALLOCATED MEMORY FOR DEVICE\n"); // UNCOMMENT if you need to see this additional information.

		// Free host memory
		free(h_neighborsOfAllVertices);
		free(h_neighborSizeArray);
		free(h_colors_found);
		//printf("DELETED ALLOCATED MEMORY FOR HOST \n"); // UNCOMMENT if you need to see this additional information.

	}
	
}
