#include <gtest/gtest.h>
#include <Eigen/Core>
#include "statistics.hpp"

using namespace lmb;

template<typename D>
auto var(const D& v) {
    auto d = (v.colwise() - v.rowwise().mean()).matrix();
    return (d * d.transpose()) / (v.cols() - 1);
}

TEST(StatistictsTests, nrand_single) {
    static const unsigned N = 500000;
    Eigen::Array<double, 1, Eigen::Dynamic> data(1, N);
    for(unsigned i = 0; i < N; ++i) {
        data(0, i) = nrand();
    }
    EXPECT_NEAR(data.mean(), 0.0, 1e-2);
    EXPECT_NEAR(var(data)[0], 1.0, 1e-2);
}

TEST(StatistictsTests, nrand_matrix) {
    static const unsigned N = 500000;
    Eigen::Array<double, 2, Eigen::Dynamic> data(2, N);
    Eigen::Matrix<double, 2, 1> m; m << 0, 0;
    Eigen::Matrix2d P; P.setIdentity();
    nrand(data, m, P);
    auto dm = data.rowwise().mean();
    auto dp = var(data);
    EXPECT_NEAR(dm(0), m(0), 1e-2);
    EXPECT_NEAR(dm(1), m(1), 1e-2);
    EXPECT_NEAR(dp(0), P(0), 1e-2);
    EXPECT_NEAR(dp(0, 0), P(0, 0), 1e-2);
    EXPECT_NEAR(dp(0, 1), P(0, 1), 1e-2);
    EXPECT_NEAR(dp(1, 0), P(1, 0), 1e-2);
    EXPECT_NEAR(dp(1, 1), P(1, 1), 1e-2);
}
