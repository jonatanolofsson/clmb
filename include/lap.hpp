// Copyright 2018 Jonatan Olofsson
#pragma once

#include <Eigen/Core>
#include <limits>

namespace lap {

using Assignment = Eigen::Matrix<int, Eigen::Dynamic, 1>;
using Dual = Eigen::Matrix<double, Eigen::Dynamic, 1>;
//static const double inf = 300000;
static const double inf = std::numeric_limits<double>::infinity();

template<typename CostMatrix, typename Assignment, typename RowDual, typename ColDual>  // NOLINT
void lap(const CostMatrix& C, Assignment& rowsol, RowDual& u, ColDual& v) {
    int N = C.rows();
    int M = C.cols();
    int loopcnt, numfree, prevnumfree, i, i0, j, j1, j2 = 0, k, low, min = 0, up;
    double h, umin, usubmin;
    Eigen::RowVectorXi free(N), collist(M), colsol(M), pred(M);
    Eigen::RowVectorXd d(M);

    rowsol.setConstant(-1);
    colsol.setConstant(-1);
    v.setZero();
    u.setZero();

    // augrowred
    for (i = 0; i < N; ++i) { free[i] = i; }
    for (j = 0; j < M; ++j) { collist[j] = j; }
    numfree = N;  // Each row is still unassigned

    for (loopcnt = 0; loopcnt < 2; ++loopcnt) {
        prevnumfree = numfree;
        numfree = 0;                                          // Number of rows unassigned in next iteration
        k = 0;                                          // Current nof free rows
        while (k < prevnumfree) {
            i = free[k++];                              // Select next unassigned row
            umin = C(i, 0) - v[0];                        // row dual for first column
            j1 = 0;                                     // Column iterator for this row
            usubmin = inf;                                   // Initialize replacement row dual
            for (j = 1; j < M; ++j) {                   // For each column (first one handled above)
                h = C(i, j) - v[j];                     // row dual for current column
                if (h < usubmin) {                           // Update only if better than second best
                    if (h >= umin) {                      // Not better than best
                        usubmin = h;                         // Set new second best - value
                        j2 = j;                         // Set new second best - column
                    } else {                            // Better than best: Set the new low as new bar
                        usubmin = umin;                        // Previous best now second best - value
                        j2 = j1;                        // Previous best now second best - column
                        umin = h;                         // Best now best - value
                        j1 = j;                         // Best now best - column
                    }
                }
            }
            i0 = colsol[j1];                                 // Get assigned row of best column
            if (umin < usubmin) {                              // If best dual is less than second best dual
                v[j1] -= usubmin - umin;                       // Update the column dual to the second best
                if (i0 >= 0) {                          // If the best column has been assigned
                    free[--k] = i0;                     // Replace the previous assignment to free list
                    rowsol[i0] = -1;                         // Unassign the row where the best column is assigned
                }
            } else if (i0 >= 0) {                       // If the second best column is equally good and the first is assigned
                j1 = j2;                                // Switch column
                i0 = colsol[j1];                             // Get assigned row of new column
                free[numfree++] = i0;                         // Set switched row as free in next iteration
                rowsol[i0] = -1;                             // Unassign switched row
            }
            rowsol[i] = j1;                                  // Set row-wise solution
            colsol[j1] = i;                                  // Set column-wise solution
        }
    }

    // --------------------------------------------------------
    // Augmentation:
    //

    for (int f = 0; f < numfree; ++f) {                          // Find augmenting path for each unassigned row
        int freerow = free[f];                                   // Index of current free row
        for (j = 0; j < M; ++j) {                       // For each column
            d[j] = C(freerow, j) - v[j];                     // Cost to move to each column
            pred[j] = freerow;                               // row-predecessor of column in augmenting/alternating path.
            //collist[j] = j; ??
        }
        low = 0;                                        // columns in 0..low-1 are ready, now none.
        up = 0;                                         // columns in low..up-1 are to be scanned for current minimum, now none.
                                                        // columns in up..dim-1 are to be considered later to find new minimum,
                                                        // at this stage the list simply contains all columns
        while (true) {
            if (up == low) {                            // Find columns with new value for minimum d
                min = d[collist[up++]];
                for (k = up; k < M; ++k) {
                    j = collist[k];
                    h = d[j];
                    if (h <= min) {
                        if (h < min) {
                            up = low;
                            min = h;
                        }
                        collist[k] = collist[up];
                        collist[up++] = j;
                    }
                }
                for (k = low; k < up; ++k) {
                    j = collist[k];
                    if (colsol[j] == -1) {
                        goto augment;
                    }
                }
            }  //{ up=low }
            j1 = collist[low++];
            i = colsol[j1];
            h = C(i, j1) - v[j1] - min;

            for (k = up; k < M; ++k) {
                j = collist[k];
                double v2 = C(i, j) - v[j] - h;
                if (v2 < d[j]) {
                    d[j] = v2;
                    pred[j] = i;
                    if ((v2 - min) < 1e-7) {
                        if (colsol[j] == -1) {
                            goto augment;
                        } else {
                            collist[k] = collist[up];
                            collist[up++] = j;
                        }
                    }
                }
            }  // { for k }
        }

        augment:
        for (k = 0; k < low; ++k) {  // Update column prices
            j1 = collist[k];
            v[j1] += d[j1] - min;
        }
        do {  // Augmentation
            i = pred[j];
            colsol[j] = i;
            k = j;
            j = rowsol[i];
            rowsol[i] = k;
        } while (i != freerow);
    }  // { for numfree }

    for (i = 0; i < N; ++i) {
        j = rowsol[i];
        u[i] = C(i, j) - v[j];
    }
}


template<typename CostMatrix>
Assignment lap(const CostMatrix& C) {
    Dual u(C.rows());
    Dual v(C.cols());
    Assignment x(C.rows());
    lap(C, x, u, v);
    return x;
}

template<typename CostMatrix>
inline double cost(CostMatrix& C, Assignment res) {
    double c = 0;
    for (unsigned i = 0; i < res.rows(); ++i) {
        c += C(i, res[i]);
    }
    return c;
}

}  // namespace lap
