#pragma once
// https://raw.githubusercontent.com/ClementLF/rlapjv/master/rlapjv.h
#include <cassert>
#include <cstdio>
#include <limits>
#include <memory>
#include <Eigen/Dense>

namespace lap {
using CostMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Assignment = Eigen::Matrix<int, Eigen::Dynamic, 1>;
using Dual = Eigen::Matrix<double, Eigen::Dynamic, 1>;
static const double inf = std::numeric_limits<double>::infinity();

#ifdef __GNUC__
#define always_inline __attribute__((always_inline)) inline
#define restrict __restrict__
#elif _WIN32
#define always_inline __forceinline
#define restrict __restrict
#else
#define always_inline inline
#define restrict
#endif

template <typename idx, typename cost>
always_inline std::tuple<cost, cost, idx, idx>
find_umins_plain(
    idx dim, idx i, const cost *restrict assign_cost,
    const cost *restrict v)
{
    const cost *local_cost = &assign_cost[i * dim];
    cost umin = local_cost[0] - v[0];
    idx j1 = 0;
    idx j2 = -1;
    cost usubmin = std::numeric_limits<cost>::infinity();
    for (idx j = 1; j < dim; j++)
    {
        cost h = local_cost[j] - v[j];
        if (h < usubmin)
        {
            if (h >= umin)
            {
                usubmin = h;
                j2 = j;
            }
            else
            {
                usubmin = umin;
                umin = h;
                j2 = j1;
                j1 = j;
            }
        }
    }
    return std::make_tuple(umin, usubmin, j1, j2);
}

// MSVC++ has an awful AVX2 support which does not allow to compile the code
#if defined(__AVX2__) && !defined(_WIN32)
#include <immintrin.h>

#define FLOAT_MIN_DIM 64
#define DOUBLE_MIN_DIM 100000 // 64-bit code is actually always slower

template <typename idx>
always_inline std::tuple<float, float, idx, idx>
find_umins(
    idx dim, idx i, const float *restrict assign_cost,
    const float *restrict v)
{
    if (dim < FLOAT_MIN_DIM)
    {
        return find_umins_plain(dim, i, assign_cost, v);
    }
    const float *local_cost = assign_cost + i * dim;
    __m256i idxvec = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i j1vec = _mm256_set1_epi32(-1), j2vec = _mm256_set1_epi32(-1);
    __m256 uminvec = _mm256_set1_ps(std::numeric_limits<float>::infinity()),
           usubminvec = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    for (idx j = 0; j < dim - 7; j += 8)
    {
        __m256 acvec = _mm256_loadu_ps(local_cost + j);
        __m256 vvec = _mm256_loadu_ps(v + j);
        __m256 h = _mm256_sub_ps(acvec, vvec);
        __m256 cmp = _mm256_cmp_ps(h, uminvec, _CMP_LE_OQ);
        usubminvec = _mm256_blendv_ps(usubminvec, uminvec, cmp);
        j2vec = _mm256_blendv_epi8(
            j2vec, j1vec, reinterpret_cast<__m256i>(cmp));
        uminvec = _mm256_blendv_ps(uminvec, h, cmp);
        j1vec = _mm256_blendv_epi8(
            j1vec, idxvec, reinterpret_cast<__m256i>(cmp));
        cmp = _mm256_andnot_ps(cmp, _mm256_cmp_ps(h, usubminvec, _CMP_LT_OQ));
        usubminvec = _mm256_blendv_ps(usubminvec, h, cmp);
        j2vec = _mm256_blendv_epi8(
            j2vec, idxvec, reinterpret_cast<__m256i>(cmp));
        idxvec = _mm256_add_epi32(idxvec, _mm256_set1_epi32(8));
    }
    alignas(__m256) float uminmem[8], usubminmem[8];
    alignas(__m256) int32_t j1mem[8], j2mem[8];
    _mm256_store_ps(uminmem, uminvec);
    _mm256_store_ps(usubminmem, usubminvec);
    _mm256_store_si256(reinterpret_cast<__m256i *>(j1mem), j1vec);
    _mm256_store_si256(reinterpret_cast<__m256i *>(j2mem), j2vec);

    idx j1 = -1, j2 = -1;
    float umin = std::numeric_limits<float>::infinity(),
          usubmin = std::numeric_limits<float>::infinity();
    for (int vi = 0; vi < 8; vi++)
    {
        float h = uminmem[vi];
        if (h < usubmin)
        {
            idx jnew = j1mem[vi];
            if (h >= umin)
            {
                usubmin = h;
                j2 = jnew;
            }
            else
            {
                usubmin = umin;
                umin = h;
                j2 = j1;
                j1 = jnew;
            }
        }
    }
    for (int vi = 0; vi < 8; vi++)
    {
        float h = usubminmem[vi];
        if (h < usubmin)
        {
            usubmin = h;
            j2 = j2mem[vi];
        }
    }
    for (idx j = dim & 0xFFFFFFF8u; j < dim; j++)
    {
        float h = local_cost[j] - v[j];
        if (h < usubmin)
        {
            if (h >= umin)
            {
                usubmin = h;
                j2 = j;
            }
            else
            {
                usubmin = umin;
                umin = h;
                j2 = j1;
                j1 = j;
            }
        }
    }
    return std::make_tuple(umin, usubmin, j1, j2);
}

template <typename idx>
always_inline std::tuple<double, double, idx, idx>
find_umins(
    idx dim, idx i, const double *restrict assign_cost,
    const double *restrict v)
{
    if (dim < DOUBLE_MIN_DIM)
    {
        return find_umins_plain(dim, i, assign_cost, v);
    }
    const double *local_cost = assign_cost + i * dim;
    __m256i idxvec = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i j1vec = _mm256_set1_epi64x(-1), j2vec = _mm256_set1_epi64x(-1);
    __m256d uminvec = _mm256_set1_pd(std::numeric_limits<double>::infinity()),
            usubminvec = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    for (idx j = 0; j < dim - 3; j += 4)
    {
        __m256d acvec = _mm256_loadu_pd(local_cost + j);
        __m256d vvec = _mm256_loadu_pd(v + j);
        __m256d h = _mm256_sub_pd(acvec, vvec);
        __m256d cmp = _mm256_cmp_pd(h, uminvec, _CMP_LE_OQ);
        usubminvec = _mm256_blendv_pd(usubminvec, uminvec, cmp);
        j2vec = _mm256_blendv_epi8(
            j2vec, j1vec, reinterpret_cast<__m256i>(cmp));
        uminvec = _mm256_blendv_pd(uminvec, h, cmp);
        j1vec = _mm256_blendv_epi8(
            j1vec, idxvec, reinterpret_cast<__m256i>(cmp));
        cmp = _mm256_andnot_pd(cmp, _mm256_cmp_pd(h, usubminvec, _CMP_LT_OQ));
        usubminvec = _mm256_blendv_pd(usubminvec, h, cmp);
        j2vec = _mm256_blendv_epi8(
            j2vec, idxvec, reinterpret_cast<__m256i>(cmp));
        idxvec = _mm256_add_epi64(idxvec, _mm256_set1_epi64x(4));
    }
    alignas(__m256d) double uminmem[4], usubminmem[4];
    alignas(__m256d) int64_t j1mem[4], j2mem[4];
    _mm256_store_pd(uminmem, uminvec);
    _mm256_store_pd(usubminmem, usubminvec);
    _mm256_store_si256(reinterpret_cast<__m256i *>(j1mem), j1vec);
    _mm256_store_si256(reinterpret_cast<__m256i *>(j2mem), j2vec);

    idx j1 = -1, j2 = -1;
    double umin = std::numeric_limits<double>::infinity(),
           usubmin = std::numeric_limits<double>::infinity();
    for (int vi = 0; vi < 4; vi++)
    {
        double h = uminmem[vi];
        if (h < usubmin)
        {
            idx jnew = j1mem[vi];
            if (h >= umin)
            {
                usubmin = h;
                j2 = jnew;
            }
            else
            {
                usubmin = umin;
                umin = h;
                j2 = j1;
                j1 = jnew;
            }
        }
    }
    for (int vi = 0; vi < 4; vi++)
    {
        double h = usubminmem[vi];
        if (h < usubmin)
        {
            usubmin = h;
            j2 = j2mem[vi];
        }
    }
    for (idx j = dim & 0xFFFFFFFCu; j < dim; j++)
    {
        double h = local_cost[j] - v[j];
        if (h < usubmin)
        {
            if (h >= umin)
            {
                usubmin = h;
                j2 = j;
            }
            else
            {
                usubmin = umin;
                umin = h;
                j2 = j1;
                j1 = j;
            }
        }
    }
    return std::make_tuple(umin, usubmin, j1, j2);
}

#else // __AVX__

#define find_umins find_umins_plain

#endif // __AVX__

/// @brief Jonker-Volgenant algorithm.
/// @param dim in problem size
/// @param assign_cost in cost matrix
/// @param verbose in indicates whether to report the progress to stdout
/// @param rowsol out column assigned to row in solution / size dim
/// @param colsol out row assigned to column in solution / size dim
/// @param u out dual variables, row reduction numbers / size dim
/// @param v out dual variables, column reduction numbers / size dim
/// @return achieved minimum assignment cost
template <typename idx, typename cost>
cost rlap(int dim0, int dim1,
         const cost *restrict assign_cost, bool verbose,
         idx *restrict rowsol, idx *restrict colsol,
         cost *restrict u, cost *restrict v)
{
    auto free = std::unique_ptr<idx[]>(new idx[dim0]);    // list of unassigned rows.
    auto collist = std::unique_ptr<idx[]>(new idx[dim1]); // list of columns to be scanned in various ways.
    auto d = std::unique_ptr<cost[]>(new cost[dim1]);     // 'cost-distance' in augmenting path calculation.
    auto pred = std::unique_ptr<idx[]>(new idx[dim1]);    // row-predecessor of column in augmenting/alternating path.
    idx numfree = 0;

#if _OPENMP >= 201307
#pragma omp simd
#endif


    for (idx i = 0; i < dim0; i++)
    {
        free[i] = i;
        rowsol[i] = -1;
        numfree++;
    }


    for (idx j = 0; j < dim1; j++)
    {
        colsol[j] = -1;
    }


    // AUGMENTING ROW REDUCTION
    for (int loopcnt = 0; loopcnt < 2; loopcnt++)
    { // loop to be done twice.
        // scan all free rows.
        // in some cases, a free row may be replaced with another one to be scanned next.
        idx k = 0;
        idx prevnumfree = numfree;
        numfree = 0; // start list of rows still free after augmenting row reduction.
        while (k < prevnumfree)
        {
            idx i = free[k++];

            // find minimum and second minimum reduced cost over columns.
            cost umin, usubmin;
            idx j1, j2;
            std::tie(umin, usubmin, j1, j2) = find_umins(dim1, i, assign_cost, v);

            idx i0 = colsol[j1];
            cost vj1_new = v[j1] - (usubmin - umin);
            bool vj1_lowers = vj1_new < v[j1]; // the trick to eliminate the epsilon bug
            if (vj1_lowers)
            {
                // change the reduction of the minimum column to increase the minimum
                // reduced cost in the row to the subminimum.
                v[j1] = vj1_new;
            }
            else if (i0 >= 0)
            { // minimum and subminimum equal.
                // minimum column j1 is assigned.
                // swap columns j1 and j2, as j2 may be unassigned.
                j1 = j2;
                i0 = colsol[j2];
            }

            // (re-)assign i to j1, possibly de-assigning an i0.
            rowsol[i] = j1;
            colsol[j1] = i;

            if (i0 >= 0)
            { // minimum column j1 assigned earlier.
                if (vj1_lowers)
                {
                    // put in current k, and go back to that k.
                    // continue augmenting path i - j1 with i0.
                    free[--k] = i0;
                }
                else
                {
                    // no further augmenting reduction possible.
                    // store i0 in list of free rows for next phase.
                    free[numfree++] = i0;
                }
            }
        }
        if (verbose)
        {
            printf("lapjv: AUGMENTING ROW REDUCTION %d / %d\n", loopcnt + 1, 2);
        }
    } // for loopcnt

    // AUGMENT SOLUTION for each free row.
    for (idx f = 0; f < numfree; f++)
    {
        idx endofpath;
        idx freerow = free[f]; // start row of augmenting path.
        if (verbose)
        {
            printf("lapjv: AUGMENT SOLUTION row %d [%d / %d]\n",
                   freerow, f + 1, numfree);
        }

// Dijkstra shortest path algorithm.
// runs until unassigned column added to shortest path tree.
#if _OPENMP >= 201307
#pragma omp simd
#endif
        for (idx j = 0; j < dim1; j++)
        {
            d[j] = assign_cost[freerow * dim1 + j] - v[j];
            pred[j] = freerow;
            collist[j] = j; // init column list.
        }

        idx low = 0; // columns in 0..low-1 are ready, now none.
        idx up = 0;  // columns in low..up-1 are to be scanned for current minimum, now none.
                     // columns in up..dim-1 are to be considered later to find new minimum,
                     // at this stage the list simply contains all columns
        bool unassigned_found = false;
        // initialized in the first iteration: low == up == 0
        idx last = 0;
        cost min = 0;
        do
        {
            if (up == low)
            { // no more columns to be scanned for current minimum.
                last = low - 1;
                // scan columns for up..dim-1 to find all indices for which new minimum occurs.
                // store these indices between low..up-1 (increasing up).
                min = d[collist[up++]];
                for (idx k = up; k < dim1; k++)
                {
                    idx j = collist[k];
                    cost h = d[j];
                    if (h <= min)
                    {
                        if (h < min)
                        {             // new minimum.
                            up = low; // restart list at index low.
                            min = h;
                        }
                        // new index with same minimum, put on undex up, and extend list.
                        collist[k] = collist[up];
                        collist[up++] = j;
                    }
                }

                // check if any of the minimum columns happens to be unassigned.
                // if so, we have an augmenting path right away.
                for (idx k = low; k < up; k++)
                {
                    if (colsol[collist[k]] < 0)
                    {
                        endofpath = collist[k];
                        unassigned_found = true;
                        break;
                    }
                }
            }

            if (!unassigned_found)
            {
                // update 'distances' between freerow and all unscanned columns, via next scanned column.
                idx j1 = collist[low];
                low++;
                idx i = colsol[j1];
                const cost *local_cost = &assign_cost[i * dim1];
                cost h = local_cost[j1] - v[j1] - min;
                for (idx k = up; k < dim1; k++)
                {
                    idx j = collist[k];
                    cost v2 = local_cost[j] - v[j] - h;
                    if (v2 < d[j])
                    {
                        pred[j] = i;
                        if (v2 == min)
                        { // new column found at same minimum value
                            if (colsol[j] < 0)
                            {
                                // if unassigned, shortest augmenting path is complete.
                                endofpath = j;
                                unassigned_found = true;
                                break;
                            }
                            else
                            { // else add to list to be scanned right away.
                                collist[k] = collist[up];
                                collist[up++] = j;
                            }
                        }
                        d[j] = v2;
                    }
                }
            }
        } while (!unassigned_found);

// update column prices.
#if _OPENMP >= 201307
#pragma omp simd
#endif
        for (idx k = 0; k <= last; k++)
        {
            idx j1 = collist[k];
            v[j1] = v[j1] + d[j1] - min;
        }

        // reset row and column assignments along the alternating path.
        {
            idx i;
            do
            {
                i = pred[endofpath];
                colsol[endofpath] = i;
                idx j1 = endofpath;
                endofpath = rowsol[i];
                rowsol[i] = j1;
            } while (i != freerow);
        }
    }
    if (verbose)
    {
        printf("lapjv: AUGMENT SOLUTION finished\n");
    }

    // calculate optimal cost.
    cost rlapcost = 0;
#if _OPENMP >= 201307
#pragma omp simd reduction(+ \
                           : rlapcost)
#endif

    for (idx i = 0; i < dim0; i++)
    {
        const cost *local_cost = &assign_cost[i * dim1];
        idx j = rowsol[i];
        if (j >= 0) {
            u[i] = local_cost[j] - v[j];
            rlapcost += local_cost[j];
        }
    }
    if (verbose)
    {
        printf("rlapjv: optimal cost calculated\n");
    }

    return rlapcost;
}

#define popcount __builtin_popcount

template<typename CostMatrix, typename Assignment, typename RowDual, typename ColDual>  // NOLINT
bool lap(const CostMatrix& C, Assignment& rowsol, RowDual& u, ColDual& v) {
    int rows = C.rows();
    int cols = C.cols();
    Eigen::VectorXi colsol(cols);
    rowsol.setZero();
    colsol.setZero();
    u.setZero();
    v.setZero();
    int neighbours;

    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> notinf =  !C.array().isInf();
    for (int n = 1; n < std::pow(2, rows); ++n) {
        neighbours = 0;
        int popsize = popcount(n);
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                if (n & (1 << row) && notinf(row, col)) {
                    ++neighbours;
                    if(neighbours < popsize) {
                        goto nextcol;
                    } else if(neighbours == popsize) {
                        goto nextn;
                    }
                }
            }
nextcol:;
        }

        //std::cout << "NOK matrix: " << std::endl;
        //std::cout << C << std::endl;
        return false;
nextn:;
    }
    //std::cout << "OK matrix: " << std::endl;
    //std::cout << C << std::endl;
    rlap<int, double>(rows, cols, C.data(), false, rowsol.data(), colsol.data(), u.data(), v.data());
    //std::cout << "rowsol: " << rowsol.transpose() << std::endl;
    //std::cout << "colsol: " << colsol.transpose() << std::endl;
    return !(rowsol.array() < 0).any();
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

}
