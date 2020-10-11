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
#ifndef VERTEX_HPP
#define VERTEX_HPP
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct Vertex {
	int vertexIndex;    	//!< each vertex has a unique id.
	int randomValue;    	//!< randomly assigned
	int count; 			    //!< number of neighbours that have higher randomValues and are uncolored yet.
	int *neighboursIndices; //!< storing the indices of the neigbours.
	int arraySize;          //!< numbers of elements inside "neighbours" array.
	int color;              //!< if -1 then it is not colored yet.
};

Vertex* createVertices(int numberOfVertices);
void settingVertexAttributes(int *startingIndexOfRows, int *columnIndices, Vertex *vertices, int numberOfVertices);
#endif


