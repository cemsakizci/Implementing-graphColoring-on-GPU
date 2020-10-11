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
#ifndef EIGEN_MTX2CSR_HPP
#define EIGEN_MTX2CSR_HPP

#include <cstdlib>
#include <fstream>
#include "lib/eigen3/Eigen/Dense"
#include "lib/eigen3/Eigen/Sparse"
#include "lib/eigen3/unsupported/Eigen/SparseExtra"

//! short-hand declaration of double to be able to switch for float.
typedef float dtype;
//! short-hand notation for a Eigen vector of variable size.
typedef Eigen::VectorXf Vec;
//! short-hand notation for an Eigen sparse matrix in row-major (C) format
typedef Eigen::SparseMatrix<dtype, Eigen::RowMajor, int > CSRMat;
//! short-hand notation for an Eigen dense matrix of variable size
typedef Eigen::MatrixXf DensMat;


/*! \brief An inline function to load an input matrix.
 *
 * \param[in,out] inMTX - General matrix which will be loaded in CSR storage format.
 * \param[in] inMTXname - The filename of the matrix that will be loaded.
 * \return bool - returns a boolean value about importing the matrix.
 *
 * This function tries to open and load the given filename into the given general matrix and returns a true value. If the operation fails, it returns false value.
 */
inline bool eigen_mtx2csr(CSRMat& inMTX, std::string inMTXname){
    std::ifstream InputFile;
    InputFile.open(inMTXname.c_str());
    if (InputFile.good()) {
        InputFile.close();
        Eigen::loadMarket<CSRMat>(inMTX, inMTXname);
        return true;
    }
    else {
        return false;
    }
}
#endif