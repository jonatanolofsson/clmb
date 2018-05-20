// Copyright 2018 Jonatan Olofsson
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <vector>

#include "gm.hpp"
#include "lmb.hpp"
#include "cf.hpp"
#include "models.hpp"
#include "sensors.hpp"

using namespace lmb;
using GaussianReport = GaussianReport_<2>;


TEST(LMBTests, ConstructLMB) {
    using Filter = SILMB<GM<4>>;
    Filter lmb;
    cf::LL origin; origin << 58.3887657, 15.6965082;
    GaussianReport::State m; m = Eigen::Vector2d({70, 10});
    GaussianReport::Covariance P; P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 3);
    z.transform_to_global(origin);
    std::vector<GaussianReport> zs({z});
    PositionSensor<Filter::Target> s;
    lmb.correct(s, zs, 1);
    ASSERT_EQ(lmb.targettree.targets.size(), 1);
    EXPECT_EQ(lmb.targettree.targets[0]->id, 0);
    EXPECT_DOUBLE_EQ(lmb.targettree.targets[0]->r, lmb.params.rB_max);
    auto mean = cf::ll2ne(lmb.targettree.targets[0]->pdf.mean(), origin);
    EXPECT_NEAR(mean[0], 70, 1e-7);
    EXPECT_NEAR(mean[1], 10, 1e-7);
}

TEST(LMBTests, RunLMB) {
    typedef SILMB<GM<4>> Tracker;
    CV<Tracker::Target> model;
    Tracker lmb;
    cf::LL origin; origin << 58.3887657, 15.6965082;
    GaussianReport z;
    z.P = Eigen::Matrix2d::Identity();
    PositionSensor<Tracker::Target> s;
    s.fov.from_gaussian(origin, 20*Eigen::Matrix2d::Identity());
    s.lambdaB = 0.5;
    double last_time = 0;
    for (double t = 0.0; t < 5; t += 1) {
        lmb.predict<CV<Tracker::Target>>(model, t, last_time);
        z.x = Eigen::Vector2d({t, t});
        z.transform_to_global(origin);
        std::vector<GaussianReport> zs({z});
        lmb.correct(s, zs, t);
        //std::cout << "\n\nTime: " << t << std::endl;
        //std::cout << "Targets: " << lmb.targettree.targets.size() << std::endl;
        //for (auto& t : lmb.targettree.targets) {
            //std::cout << "\n\tTarget " << t->id << " (" << t << ")" << std::endl;
            //std::cout << "\t\tWeight: " << t->r << std::endl;
            //std::cout << "\t\tMean: " << t->pdf.mean().transpose() << std::endl;
            //std::cout << "\t\tPDF: " << t->pdf << std::endl;
            //std::cout << "\t\tCov: " << std::endl << t->pdf.cov() << std::endl;
        //}
        last_time = t;
    }
    EXPECT_EQ(lmb.targettree.targets.size(), 1);
}

TEST(LMBTests, PHD) {
    using Filter = SILMB<GM<4>>;
    Filter lmb;
    cf::LL origin; origin << 58.3887657, 15.6965082;
    GaussianReport::State m; m = Eigen::Vector2d({0, 0});
    GaussianReport::Covariance P; P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 3);
    z.transform_to_global(origin);
    std::vector<GaussianReport> zs({z});
    PositionSensor<Filter::Target> s;
    lmb.correct(s, zs, 1);

    Eigen::Vector2d gridsize(-5, 5);
    Eigen::Matrix<double, 2, Eigen::Dynamic> points(2, int(200*200 / std::abs(gridsize.x() * gridsize.y())));
    unsigned i = 0;
    for (int x = 100; x > -100; x+=gridsize.x()) {
        for (int y = -100; y < 100; y+=gridsize.y()) {
            points.col(i++) = cf::ne2ll(cf::NE(x, y), origin);
        }
    }

    Eigen::Matrix<double, 1, Eigen::Dynamic> res = lmb.pos_phd(points, gridsize);

    EXPECT_NEAR(res.sum(), lmb.params.rB_max, 1e-1);
}

TEST(LMBTests, OSPA) {
    using Filter = SILMB<GM<4>>;
    Filter lmb;
    cf::LL origin; origin << 58.3887657, 15.6965082;
    GaussianReport::State m; m.setZero();
    GaussianReport::Covariance P; P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 3);
    z.transform_to_global(origin);
    std::vector<GaussianReport> zs({z});
    double c = 1;
    double p = 2;
    typename Filter::TargetStates truth1{{0, 0, 0, 0}};
    typename Filter::TargetStates truth2{{1, 0, 0, 0}};
    typename Filter::TargetStates truth3{{1, 0, 0, 0}, {0, 0, 0, 0}};
    typename Filter::TargetStates truth4{{0, 0, 0, 0}, {0, 0, 1, 0}};
    typename Filter::TargetStates truth5{};
    typename Filter::TargetStates truth6{{0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}};
    cf::ne2ll_i(truth1[0], origin);
    cf::ne2ll_i(truth2[0], origin);
    cf::ne2ll_i(truth3[0], origin);
    cf::ne2ll_i(truth3[1], origin);
    cf::ne2ll_i(truth4[0], origin);
    cf::ne2ll_i(truth4[1], origin);
    cf::ne2ll_i(truth6[0], origin);
    cf::ne2ll_i(truth6[1], origin);
    cf::ne2ll_i(truth6[2], origin);
    cf::ne2ll_i(truth6[3], origin);

    PositionSensor<Filter::Target> s;
    ASSERT_EQ(lmb.targettree.targets.size(), 0);
    EXPECT_DOUBLE_EQ(lmb.ospa(truth1, c, p), c);
    EXPECT_DOUBLE_EQ(lmb.ospa(truth2, c, p), c);
    EXPECT_DOUBLE_EQ(lmb.ospa(truth6, c, p), c);
    EXPECT_DOUBLE_EQ(lmb.ospa(truth1, c, p), lmb.ospa(truth4, c, p));

    lmb.correct(s, zs, 0);

    ASSERT_EQ(lmb.targettree.targets.size(), 1);
    EXPECT_DOUBLE_EQ(lmb.ospa(truth1, c, p), 0);
    EXPECT_DOUBLE_EQ(lmb.ospa(truth2, c, p), 1);
    EXPECT_DOUBLE_EQ(lmb.ospa(truth3, c, p), lmb.ospa(truth4, c, p));
    EXPECT_DOUBLE_EQ(lmb.ospa(truth2, c, p), lmb.ospa(truth5, c, p));
}

TEST(LMBTests, GOSPA) {
    using Filter = SILMB<GM<4>>;
    Filter lmb;
    cf::LL origin; origin << 58.3887657, 15.6965082;
    GaussianReport::State m; m.setZero();
    GaussianReport::Covariance P; P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 3);
    z.transform_to_global(origin);
    std::vector<GaussianReport> zs({z});
    double c = 1;
    double p = 1;
    typename Filter::TargetStates truth1{{0, 0, 0, 0}};
    typename Filter::TargetStates truth2{{1, 0, 0, 0}};
    typename Filter::TargetStates truth3{{1, 0, 0, 0}, {0, 0, 0, 0}};
    typename Filter::TargetStates truth4{{0, 0, 0, 0}, {0, 0, 1, 0}};
    typename Filter::TargetStates truth5{};
    typename Filter::TargetStates truth6{{0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}};
    cf::ne2ll_i(truth1[0], origin);
    cf::ne2ll_i(truth2[0], origin);
    cf::ne2ll_i(truth3[0], origin);
    cf::ne2ll_i(truth3[1], origin);
    cf::ne2ll_i(truth4[0], origin);
    cf::ne2ll_i(truth4[1], origin);
    cf::ne2ll_i(truth6[0], origin);
    cf::ne2ll_i(truth6[1], origin);
    cf::ne2ll_i(truth6[2], origin);
    cf::ne2ll_i(truth6[3], origin);

    PositionSensor<Filter::Target> s;
    ASSERT_EQ(lmb.targettree.targets.size(), 0);
    EXPECT_DOUBLE_EQ(lmb.gospa(truth1, c, p), c / 2);
    EXPECT_DOUBLE_EQ(lmb.gospa(truth2, c, p), c / 2);
    EXPECT_DOUBLE_EQ(lmb.gospa(truth6, c, p), 2 * c);
    EXPECT_LT(lmb.gospa(truth1, c, p), lmb.gospa(truth4, c, p));

    lmb.correct(s, zs, 0);

    ASSERT_EQ(lmb.targettree.targets.size(), 1);
    EXPECT_DOUBLE_EQ(lmb.gospa(truth1, c, p), 0);
    EXPECT_DOUBLE_EQ(lmb.gospa(truth2, c, p), 1);
    EXPECT_DOUBLE_EQ(lmb.gospa(truth3, c, p), lmb.gospa(truth4, c, p));
    EXPECT_GT(lmb.gospa(truth2, c, p), lmb.gospa(truth5, c, p));
}
