#include <gtest/gtest.h>
#include <Eigen/Core>
#include "gm.hpp"
#include "target.hpp"

using namespace lmb;

TEST(GMTests, Correct) {
    typedef GM<4> PDF;
    typedef Target<PDF> Target;
    PDF gm({0, 0, 0, 0}, PDF::Covariance::Identity());
    EXPECT_EQ(gm.c.size(), 1);
    Report::Measurement m(2); m = Eigen::Vector2d({1, 1});
    Report::Covariance P(2, 2); P = Eigen::Matrix2d::Identity();
    Report z(m, P, 1);
    PositionSensor<Target> s;
    gm.correct(z, s);
    EXPECT_EQ(gm.c.size(), 1);
    EXPECT_FLOAT_EQ(gm.c[0].w, 1.0);
    EXPECT_FLOAT_EQ(gm.c[0].m[0], 0.5);
    EXPECT_FLOAT_EQ(gm.c[0].m[1], 0.5);
    EXPECT_FLOAT_EQ(gm.eta, 0.048266176);
}

TEST(GMTests, Copy) {
    typedef GM<4> PDF;
    PDF gm({0, 0, 0, 0}, PDF::Covariance::Identity());
    PDF gm2(gm);
    PDF gm3;
    gm3 = gm;
    EXPECT_EQ(gm.c.size(), 1);
    gm += gm2;
    EXPECT_EQ(gm.c.size(), 2);
    EXPECT_EQ(gm2.c.size(), 1);
    EXPECT_EQ(gm3.c.size(), 1);
}
