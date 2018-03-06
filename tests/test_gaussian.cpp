// Copyright 2018 Jonatan Olofsson
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <vector>
#include "gm.hpp"
#include "target.hpp"
#include "cf.hpp"

using namespace lmb;
using GaussianReport = GaussianReport_<2>;

TEST(GaussianTests, Correct) {
    using PDF = Gaussian_<4>;
    using Target = Target_<PDF>;
    PDF pdf({0, 0, 0, 0}, PDF::Covariance::Identity(), 1.0);
    auto mr = pdf.mean();
    auto Pr = pdf.cov();
    EXPECT_NEAR(mr[0], 0, 1e-2);
    EXPECT_NEAR(mr[1], 0, 1e-2);
    EXPECT_NEAR(Pr(0, 0), Pr(1, 1), 1e-2);
    EXPECT_NEAR(Pr(0, 0), 1.0, 1e-2);

    GaussianReport::State m; m = Eigen::Vector2d({1, 1});
    GaussianReport::Covariance P; P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 1);
    PositionSensor<Target> s;
    cf::LL origin; origin << 0, 0;
    pdf.correct(z, s, origin);
    mr = pdf.mean();
    Pr = pdf.cov();
    EXPECT_NEAR(mr[0], 0.5, 1e-2);
    EXPECT_NEAR(mr[1], 0.5, 1e-2);
    EXPECT_NEAR(Pr(0, 0), Pr(1, 1), 1e-2);
    EXPECT_NEAR(Pr(0, 0), 0.5, 1e-2);
    EXPECT_NEAR(Pr(2, 2), Pr(3, 3), 1e-2);
    EXPECT_NEAR(Pr(2, 2), 1, 1e-2);
}

TEST(GaussianTests, pos_pdf) {
    using PDF = Gaussian_<4>;
    PDF pdf({0, 0, 0, 0}, PDF::Covariance::Identity(), 1.0);
    Eigen::Matrix<double, 2, 2> points;
    points << 0, 1,
              0, 0;

    Eigen::Matrix<double, 1, 2> res; res.setZero();
    pdf.sampled_pos_pdf(points, res);
    EXPECT_DOUBLE_EQ(res[0], 0.6224593312018545932);
    EXPECT_DOUBLE_EQ(res[1], 0.37754066879814546231);
}
