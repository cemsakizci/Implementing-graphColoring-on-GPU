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
	printf("num_vertices %d, num_edges %d\n", m, nnz);
	vector<set<int> > svector;
	set<int> s;
	for (int i = 0; i < m; i++)
		svector.push_back(s);
	int dst, src;
	for (int i = 0; i < nnz; i++) {
		getline(cfile, str);
		sscanf(str.c_str(), "%d %d", &dst, &src);

		// Bu if'i sonradan ekledim.
		if(dst != src) {
			dst--;
			src--;

			svector[src].insert(dst);
			//svector[dst].insert(src); // directed graph'larda (1,2) varken (2,1)'de var. Onun için bu satır gereksiz.
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
		printf("The graph is not symmetric\n");
		printf(">>>>>>>>>>>>>nnz is %d, count is %d\n", nnz, count);
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
	printf("mindeg %d maxdeg %d avgdeg %.2f variance %.2f\n", mindeg, maxdeg, avgdeg, variance);
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