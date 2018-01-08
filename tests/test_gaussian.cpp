#include <gtest/gtest.h>
#include <Eigen/Core>
#include "gm.hpp"
#include "target.hpp"

using namespace lmb;

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
    EXPECT_NEAR(Pr(0, 0), 0.5, 1e-2);
    EXPECT_NEAR(Pr(2, 2), Pr(3, 3), 1e-2);
    EXPECT_NEAR(Pr(2, 2), 1, 1e-2);
    std::cout << Pr << std::endl;
}
