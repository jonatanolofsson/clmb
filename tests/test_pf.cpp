#include <gtest/gtest.h>
#include <Eigen/Core>
#include "pf.hpp"
#include "target.hpp"

using namespace lmb;

TEST(PFTests, Correct) {
    typedef PF<4, 500000> PDF;
    typedef Target<PDF> Target;
    PDF pf({0, 0, 0, 0}, PDF::Covariance::Identity());
    auto mr = pf.mean();
    auto Pr = pf.cov();
    EXPECT_NEAR(pf.w.sum(), 1.0, 1e-2);
    EXPECT_NEAR(mr[0], 0, 1e-2);
    EXPECT_NEAR(mr[1], 0, 1e-2);
    EXPECT_NEAR(Pr(0, 0), Pr(1, 1), 1e-2);
    EXPECT_NEAR(Pr(0, 0), 1.0, 1e-2);

    Report::Measurement m(2); m = Eigen::Vector2d({1, 1});
    Report::Covariance P(2, 2); P = Eigen::Matrix2d::Identity();
    Report z(m, P, 1);
    PositionSensor<Target> s;
    pf.correct(z, s);
    mr = pf.mean();
    Pr = pf.cov();
    EXPECT_NEAR(pf.w.sum(), 1.0, 1e-2);
    EXPECT_NEAR(mr[0], 0.5, 1e-2);
    EXPECT_NEAR(mr[1], 0.5, 1e-2);
    EXPECT_NEAR(Pr(0, 0), Pr(1, 1), 1e-2);
}

TEST(PFTests, Copy) {
    typedef PF<2, 100> PDF;
    Eigen::Matrix<double, 2, 1> mean; mean << 0, 0;
    auto P = PDF::Covariance::Identity();
    PDF pf(mean, P);
    PDF pf2(pf);
    PDF pf3;
    pf3 = pf;
    EXPECT_EQ(pf.w.size(), 100);
    EXPECT_EQ(pf2.w.size(), 100);
    EXPECT_EQ(pf3.w.size(), 100);
}
