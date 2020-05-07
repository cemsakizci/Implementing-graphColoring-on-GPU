#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct vertex {
	int vertexIndex;// each vertex has a unique id.
	int randomValue; // randomly assigned
	int count; // number of neighbours that have higher randomValues and are uncolored yet.
	int *neighboursIndices; // storing the indices of the neigbours.
	int arraySize; // numbers of elements inside "neighbours" array.
	int color; // if -1 then it is not colored yet.
};

typedef struct vertex Vertex;

Vertex* createVertices(int numberOfVertices);
void settingVertexAttributes(int *startingIndexOfRows, int *columnIndices, Vertex *vertices, int numberOfVertices);
//int coloringAllVertices(Vertex *vertices, int numberOfVertices, int checkingStillUncolored);



