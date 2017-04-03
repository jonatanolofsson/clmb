#include "hypgen.hpp"
#include "lapjv/lap.h"

namespace lmb {
    //Murty::Murty(const CostMatrix C) {
        //size = C.cols();
        //state.C = C;
    //}

    //void Murty::draw(Assignment&, double&) {
    //}

    void lapjv(const CostMatrix& cost, Assignment& res, double& total_cost) {
        unsigned N = cost.rows();
        unsigned M = cost.cols();

        int* colsol = new int[M];
        double* u = new double[M];
        double* v = new double[M];
        unsigned i;

        auto c = new double*[M];
        for (i = 0; i < N; ++i) {
            c[i] = new double[M];
            for (unsigned j = 0; j < M; ++j) {
                c[i][j] = cost(i, j);
            }
        }
        for (i = N; i < M; ++i) {
            c[i] = new double[M];
            for (unsigned j = 0; j < M; ++j) {
                c[i][j] = 0.0;
            }
        }

        total_cost = ::lap(M, c, res.data(), colsol, u, v);

        for (i = 0; i < M; ++i) {
            delete c[i];
        }
        delete[] c;
        delete[] colsol;
        delete[] u;
        delete[] v;
    }
}
