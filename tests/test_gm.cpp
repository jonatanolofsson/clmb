// Copyright 2018 Jonatan Olofsson
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <vector>
#include "gm.hpp"
#include "target.hpp"
#include "params.hpp"

using namespace lmb;
using GaussianReport = GaussianReport_<2>;

TEST(GMTests, Correct) {
    using PDF = GM<4>;
    using Target = Target_<PDF>;
    Params params;
    PDF pdf(&params, {0, 0, 0, 0}, PDF::Covariance::Identity());
    auto mr = pdf.mean();
    auto Pr = pdf.cov();
    EXPECT_NEAR(mr[0], 0, 1e-2);
    EXPECT_NEAR(mr[1], 0, 1e-2);
    EXPECT_NEAR(Pr(0, 0), Pr(1, 1), 1e-2);
    EXPECT_NEAR(Pr(0, 0), 1.0, 1e-2);
    EXPECT_EQ(pdf.c.size(), 1);

    GaussianReport::State m; m = Eigen::Vector2d({1, 1});
    GaussianReport::Covariance P; P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 1);
    PositionSensor<Target> s;
    s.fov = BBox();
    pdf.correct(z, s);
    mr = pdf.mean();
    Pr = pdf.cov();
    EXPECT_NEAR(mr[0], 0.5, 1e-2);
    EXPECT_NEAR(mr[1], 0.5, 1e-2);
    EXPECT_NEAR(Pr(0, 0), Pr(1, 1), 1e-2);
    EXPECT_NEAR(pdf.eta, 0.05, 5e-2);
    EXPECT_EQ(pdf.c.size(), 1);
}

TEST(GMTests, Copy) {
    using PDF = GM<2>;
    Params params;
    PDF pdf(&params, {0, 0}, PDF::Covariance::Identity());
    PDF pdf2(pdf);
    PDF pdf3(&params);
    pdf3 = pdf;
    EXPECT_EQ(pdf.c.size(), 1);
    pdf2 += pdf3;
    EXPECT_EQ(pdf.c.size(), 1);
    EXPECT_EQ(pdf2.c.size(), 2);
    EXPECT_EQ(pdf3.c.size(), 1);
    pdf2.clear();
    EXPECT_EQ(pdf.c.size(), 1);
    EXPECT_EQ(pdf2.c.size(), 0);
}

TEST(GMTests, pos_pdf) {
    using PDF = GM<2>;
    Params params;
    PDF pdf1(&params, {0, 0}, PDF::Covariance::Identity());
    PDF pdf2(&params, {1, 0}, PDF::Covariance::Identity());
    pdf2 += pdf1;
    Eigen::Array<double, 2, Eigen::Dynamic> points(2, 2);
    points << 0, 1,
              0, 0;

    Eigen::Array<double, 1, Eigen::Dynamic> res(1, 2); res.setZero();
    pdf2.sampled_pos_pdf(points, res);
    EXPECT_DOUBLE_EQ(res[0], 0.5);
    EXPECT_DOUBLE_EQ(res[1], 0.5);
}
