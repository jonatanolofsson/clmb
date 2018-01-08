#include <gtest/gtest.h>
#include <Eigen/Core>
#include "gm.hpp"
#include "target.hpp"
#include "params.hpp"

using namespace lmb;

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

    GaussianReport::Measurement m(2); m = Eigen::Vector2d({1, 1});
    GaussianReport::Covariance P(2, 2); P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 1);
    PositionSensor<Target> s;
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
    pdf += pdf2;
    EXPECT_EQ(pdf.c.size(), 2);
    EXPECT_EQ(pdf2.c.size(), 1);
    EXPECT_EQ(pdf3.c.size(), 1);
}
