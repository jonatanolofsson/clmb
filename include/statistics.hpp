#pragma once
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace lmb {
    double urand();
    double nrand();

    template<typename RES, typename MEAN, typename COV>
    void nrand(RES& res, const MEAN mean, const COV covariance) {
        Eigen::SelfAdjointEigenSolver<COV> eigenSolver(covariance);
        auto transform = eigenSolver.eigenvectors() \
            * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
        for(Eigen::DenseIndex i = 0; i < res.cols(); ++i) {
            res.col(i) = mean + transform * MEAN{}.unaryExpr(
                [&](auto) { return nrand(); } );
        }
    }
}
