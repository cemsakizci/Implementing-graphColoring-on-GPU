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

#include"vertex.h"

Vertex* createVertices(int numberOfVertices) {
	srand(time(NULL));
	Vertex *vertices = (Vertex *) malloc(numberOfVertices * sizeof(Vertex));
	int currentIndex = 0;
	
	int *randomValues = (int *) malloc((numberOfVertices-1) * sizeof(int)); // storing generated random values so that vertices have different random values.

	for(int i=0; i<numberOfVertices-1; i++) 
		randomValues[i] = 0; // Initially every entry is 0.
		
	for(int i=0; i<numberOfVertices; i++) {
		vertices[i].vertexIndex = currentIndex++;

		// we have the following do-while to prevent having the same random values.
		int havingRandomValue = 0, upperLimit = numberOfVertices*5; // To make the assignment of random values easy, enlarge the range.
		while(!havingRandomValue) { // it will loop here until it gets a random value.
			int currentRandomValue = (rand() % upperLimit) + 1; // the limit is from 1 to upperLimit.
			for(int j=0; j<numberOfVertices-1; j++) { // the reason for "numberOfVertices-1" is that we don't have to store the color of the last vertex to check.
				if(randomValues[j] == 0 || (j == numberOfVertices-2)) { // the second logical expression checks whether it has reached the random value of the last vertex or not.
					if(randomValues[j] == 0)
						randomValues[j] = currentRandomValue;
					else if(randomValues[numberOfVertices-2] == currentRandomValue)
						break;
					vertices[i].randomValue = currentRandomValue;
					havingRandomValue = 1;
					break;
				}
				else if(randomValues[j] == currentRandomValue) // checking whether a currentRandomValue has assigned previously or not.
					break;
			}
			
		}

		vertices[i].count = 0;
		vertices[i].neighboursIndices = NULL;
		vertices[i].arraySize = 0;
		vertices[i].color = -1;
		//printf("V%d --> %d\n", i, vertices[i].randomValue); // to see the random value of each vertex.
	}
	
	free(randomValues);

	return vertices;	
}

void settingVertexAttributes(int *startingIndexOfRows, int *columnIndices, Vertex *vertices, int numberOfVertices) {

	int numberOfNeighbours = 0, columnTraceIndex = 0, randomValueOfCurrentVertex;
	for(int i=0; i<numberOfVertices; i++) {

		numberOfNeighbours = startingIndexOfRows[i+1] - startingIndexOfRows[i]; // how many neighbors there are for i.th vertex.
		vertices[i].neighboursIndices = (int *) malloc(numberOfNeighbours * sizeof(int)); // allocating memory to hold neighbour indices
		vertices[i].arraySize = numberOfNeighbours;	// setting array size of the current vertex
		randomValueOfCurrentVertex = vertices[i].randomValue; // local variable for random value of the current vertex

		for(int j=0; j<numberOfNeighbours; j++) { // tracing in the neighbours of the current vertex

			int indexOfTheNeighbour = columnIndices[columnTraceIndex]; // finding the index of the specific neighbour
			columnTraceIndex++;
			(vertices[i].neighboursIndices)[j] = indexOfTheNeighbour; // adding that neighbour's index to the current vertex's neighbourIndices array.
			Vertex neighbour = vertices[indexOfTheNeighbour]; // local variable which stores this neighbour
			
			// checking if the neighbour has a greater random value and is uncolored yet.
			if((randomValueOfCurrentVertex < neighbour.randomValue) && neighbour.color == -1) 
				vertices[i].count++;
		}
	}

}
