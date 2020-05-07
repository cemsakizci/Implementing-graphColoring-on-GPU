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
		int havingRandomValue = 0;
		while(!havingRandomValue) { // random value yok iken, while'da dönecek.
			int currentRandomValue = (rand() % 200) + 1; // 1'den 200'e kadar sınır.
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
		printf("V%d --> %d\n", i, vertices[i].randomValue);
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

/*
// sorting'te hata var gibi ---> (93. satır) if(vertices[nextNeighbourIndex].color < vertices[currentVertex.neighboursIndices[minColorIndex]].color)
int coloringAllVertices(Vertex *vertices, int numberOfVertices, int checkingStillUncolored) {

	for(int i=0; i<numberOfVertices; i++) {

		Vertex currentVertex = vertices[i];
		printf("vertexID:%d count:%d\n", i, currentVertex.count);
		if(currentVertex.count == 0) { // says that it is time to color this vertex.
			if(currentVertex.color == -1) { // we skip the colored vertices.
				
				// Finding the min color that the neighbours of the current vertex don't have 
				
				// we sort the neighboursIndices array according to the their colors.
				for(int k=0; k < currentVertex.arraySize-1; k++) {
					int currentNeighbourIndex = currentVertex.neighboursIndices[k];
					
					int minColorIndex = k; // suppose that the vertex having neighboursIndices[k] index(for the current vertex) has the minimum color at the beginning.
					for(int t=k+1; t < currentVertex.arraySize; t++) { // this loop compares the kth vertex's color with the right hand side of the neighbourIndices array (by starting at t).
						int nextNeighbourIndex = currentVertex.neighboursIndices[t];
						if(vertices[nextNeighbourIndex].color < vertices[currentVertex.neighboursIndices[minColorIndex]].color)
							minColorIndex = t; // changing the minimum color to the vertex which has the following index --> neighboursIndices[t](for the current vertex).
					}
					if(minColorIndex != k) {
						currentVertex.neighboursIndices[k] = currentVertex.neighboursIndices[minColorIndex];
						currentVertex.neighboursIndices[minColorIndex] = currentNeighbourIndex;
					}
				}

				// coloring the vertex.
				int secondNeighbourIndex;
				for(int k=0; k<currentVertex.arraySize-1; k++) {
					int firstNeighbourIndex = currentVertex.neighboursIndices[k];
					if(vertices[firstNeighbourIndex].color > 0 && k==0) { // if the 0th neighbour of the current vertex has the color which is greater than 0, then color the current vertex with 0.
						currentVertex.color = 0;
						printf("coloring --- vertexID:%d color:%d\n", i, currentVertex.color);
						break;
					}
					secondNeighbourIndex = currentVertex.neighboursIndices[k+1];			
					if((vertices[secondNeighbourIndex].color - vertices[firstNeighbourIndex].color) > 1) {
						printf("index %d\n", currentVertex.vertexIndex);	
						currentVertex.color = vertices[firstNeighbourIndex].color + 1;
						printf("coloring --- vertexID:%d color:%d\n", i, currentVertex.color);
						break;
					}
					
				}
				if(currentVertex.color == -1) { // if the vertex is still uncolored(ex : in the case of 4 neighbours with colors --> [0,1,2,3]), then color the vertex with the color of the last neighbour + 1.
					
					currentVertex.color = vertices[secondNeighbourIndex].color + 1;
					printf("coloring --- vertexID:%d color:%d\n", i, currentVertex.color);
			
				}
				
				// decrement each neighbour's count by 1
				for(int j=0; j<currentVertex.arraySize; j++) { 
					int neighbourIndex = currentVertex.neighboursIndices[j];
					if(vertices[neighbourIndex].count > 0)
						vertices[neighbourIndex].count--;
				}
			vertices[i] = currentVertex; // reflect the changes, which have been made to the local variable "currentVertex", to vertices[i].
			}
		}
		else
			checkingStillUncolored = 1; // checking if there is any vertex that needs to be colored.	

	}
	
	return checkingStillUncolored;

}
*/
