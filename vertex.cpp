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
		while(!havingRandomValue) { // random value yok iken, while'da dönecek.
			int currentRandomValue = (rand() % upperLimit) + 1; // 1'den upperLimit'e kadar sınır.
			for(int j=0; j<numberOfVertices-1; j++) { // "numberOfVertices-1" olmasının sebebi : sonuncu vertex'in rengini tutmaya gerek yok kontrol etmek için.
				if(randomValues[j] == 0 || (j == numberOfVertices-2)) { // 2. logical expression sonuncu vertex'in random değerine mi geldi ona bakıcak.
					if(randomValues[j] == 0)
						randomValues[j] = currentRandomValue;
					else if(randomValues[numberOfVertices-2] == currentRandomValue)
						break;
					vertices[i].randomValue = currentRandomValue;
					havingRandomValue = 1;
					break;
				}
				else if(randomValues[j] == currentRandomValue) // currentRandomValue önceden atanmış mı onu kontrol ediyor.
					break;
			}
			
		}

		vertices[i].count = 0;
		vertices[i].neighboursIndices = NULL;
		vertices[i].arraySize = 0;
		vertices[i].color = -1;
		//printf("V%d --> %d\n", i, vertices[i].randomValue);
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
