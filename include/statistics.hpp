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

#include <random>
#include <Eigen/Eigenvalues>
#include "statistics.hpp"

namespace lmb {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    static std::uniform_real_distribution<> udist;
    static std::normal_distribution<> ndist;

    double urand() {
        return udist(gen);
    }

    double nrand() {
        return ndist(gen);
    }
}
