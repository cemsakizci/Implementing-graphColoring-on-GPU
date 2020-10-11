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

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <vector>
#include <set>

#include "mtx2csr.h"

using namespace std;

void mtx2csr(char *mtx, int &m, int &nnz, int *&csrRowPtr, int *&csrColInd) {
	printf("Reading .mtx input file %s\n", mtx);
	std::ifstream cfile;
	cfile.open(mtx);
	std::string str;
	getline(cfile, str);
	char c;
	sscanf(str.c_str(), "%c", &c);
	while (c == '%') {
		getline(cfile, str);
		sscanf(str.c_str(), "%c", &c);
	}
	int n;
	sscanf(str.c_str(), "%d %d %d", &m, &n, &nnz);
	if (m != n) {
		printf("error!\n");
		exit(0);
	}
	//printf("num_vertices %d, num_edges %d\n", m, nnz); // UNCOMMENT if you need to see this additional information.
	vector<set<int> > svector;
	set<int> s;
	for (int i = 0; i < m; i++)
		svector.push_back(s);
	int dst, src;
	for (int i = 0; i < nnz; i++) {
		getline(cfile, str);
		sscanf(str.c_str(), "%d %d", &dst, &src);

		// since the same vertex cannot be a neighbor of itself, we make the distinction inside this if statement.
		if(dst != src) {
			dst--;
			src--;

			svector[src].insert(dst);
			/* For undirected graphs, when there is an occurence of (1,2), we have to add (2,1) as well because it is not explicitly available in the graph. */
			svector[dst].insert(src);
		}
		
	}
	cfile.close();
	csrRowPtr = (int *)malloc((m + 1) * sizeof(int));
	int count = 0;
	for (int i = 0; i < m; i++) {
		csrRowPtr[i] = count;
		count += svector[i].size();
	}
	csrRowPtr[m] = count;
	if (count != nnz) {
		//printf("The graph is not symmetric\n"); // UNCOMMENT if you need to see this additional information.
		//printf(">>>>>>>>>>>>>nnz is %d, count is %d\n", nnz, count); // UNCOMMENT if you need to see this additional information.
		nnz = count;
		
	}
	
	/*
	// To check inside of csrRowPtr
	printf("------start of csrRowPtr---------\n");
	for(int i=0; i<m+1; i++) {
		printf("%d, ", csrRowPtr[i]);
	}
	printf("\n");
	printf("------end of csrRowPtr-------\n");
	*/

	double avgdeg;
	double variance = 0.0;
	int maxdeg = 0;
	int mindeg = m;
	avgdeg = (double)nnz / m;
	for (int i = 0; i < m; i++) {
		int deg_i = csrRowPtr[i + 1] - csrRowPtr[i];
		if (deg_i > maxdeg)
			maxdeg = deg_i;
		if (deg_i < mindeg)
			mindeg = deg_i;
		variance += (deg_i - avgdeg) * (deg_i - avgdeg) / m;
	}
	//printf("mindeg %d maxdeg %d avgdeg %.2f variance %.2f\n", mindeg, maxdeg, avgdeg, variance); // UNCOMMENT if you need to see this additional information.
	csrColInd = (int *)malloc(count * sizeof(int));
	set<int>::iterator site;
	for (int i = 0, index = 0; i < m; i++) {
		site = svector[i].begin();
		while (site != svector[i].end()) {
			csrColInd[index++] = *site;
			site++;
		}
	}
	
	/*
	// to check inside of csrColInd
	printf("-----start of csrColInd--------\n");
	for(int i=0; i<count; i++) {
		printf("%d, ", csrColInd[i]);
	}
	printf("\n");
	printf("--------end of csrColInd-------\n");
	*/
}

/*
int main(int argc, char *argv[]) {
	if (argc != 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}

	int m, nnz, *csrRowPtr = NULL, *csrColInd = NULL;
	if (strstr(argv[1], ".mtx"))
		mtx2csr(argv[1], m, nnz, csrRowPtr, csrColInd);
	else {
		printf("Unrecognizable input file format\n");
		exit(0);
	}

	if (csrRowPtr == NULL)
		printf("csrRowPtr is NULL\n");
	if (csrColInd == NULL)
		printf("csrColInd is NULL\n");
	
	return 0;
	
}
*/
