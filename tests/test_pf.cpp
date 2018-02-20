#include <gtest/gtest.h>
#include <Eigen/Core>
#include "pf.hpp"
#include "target.hpp"
#include "params.hpp"

using namespace lmb;
using GaussianReport = GaussianReport_<2>;

TEST(PFTests, Correct) {
    using PDF = PF<4, 500000>;
    using Target = Target_<PDF>;
    Params params;
    PDF pdf(&params, {0, 0, 0, 0}, PDF::Covariance::Identity());
    auto mr = pdf.mean();
    auto Pr = pdf.cov();
    EXPECT_NEAR(mr[0], 0, 1e-2);
    EXPECT_NEAR(mr[1], 0, 1e-2);
    EXPECT_NEAR(Pr(0, 0), Pr(1, 1), 1e-2);
    EXPECT_NEAR(Pr(0, 0), 1.0, 1e-2);
    EXPECT_NEAR(pdf.w.sum(), 1.0, 1e-2);

    GaussianReport::State m; m = Eigen::Vector2d({1, 1});
    GaussianReport::Covariance P; P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 1);
    PositionSensor<Target> s;
    pdf.correct(z, s);
    mr = pdf.mean();
    Pr = pdf.cov();
    EXPECT_NEAR(mr[0], 0.5, 1e-2);
    EXPECT_NEAR(mr[1], 0.5, 1e-2);
    EXPECT_NEAR(Pr(0, 0), Pr(1, 1), 1e-2);
    EXPECT_NEAR(pdf.eta, 0.05, 5e-2);
    EXPECT_NEAR(pdf.w.sum(), 1.0, 1e-2);
}

TEST(PFTests, Copy) {
    typedef PF<2, 100> PDF;
    Params params;
    PDF pdf(&params, {0, 0}, PDF::Covariance::Identity());
    PDF pdf2(pdf);
    PDF pdf3;
    pdf3 = pdf;
    EXPECT_EQ(pdf.w.size(), 100);
    EXPECT_EQ(pdf2.w.size(), 100);
    EXPECT_EQ(pdf3.w.size(), 100);
}
