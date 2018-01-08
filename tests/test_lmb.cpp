#include <Eigen/Core>
#include <gtest/gtest.h>
#include <vector>

#include "gm.hpp"
#include "lmb.hpp"
#include "models.hpp"
#include "sensors.hpp"

using namespace lmb;


TEST(LMBTests, ConstructLMB) {
    using Filter = SILMB<GM<4>>;
    Filter lmb;
    GaussianReport::Measurement m(2); m = Eigen::Vector2d({1, 1});
    GaussianReport::Covariance P(4, 4); P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 3);
    std::vector<GaussianReport> zs({z});
    PositionSensor<Filter::Target> s;
    lmb.correct(zs, s, 1);
    ASSERT_EQ(lmb.targettree.targets.size(), 1);
    EXPECT_EQ(lmb.targettree.targets[0]->id, 0);
    EXPECT_FLOAT_EQ(lmb.targettree.targets[0]->r, lmb.params.rB_max);
    auto mean = lmb.targettree.targets[0]->pdf.mean();
    EXPECT_FLOAT_EQ(mean[0], 1);
    EXPECT_FLOAT_EQ(mean[1], 1);
}

TEST(LMBTests, RunLMB) {
    using Filter = SILMB<GM<4>>;
    Filter lmb;
    GaussianReport::Measurement m(2); m.setZero();
    GaussianReport::Covariance P(4, 4); P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 3);
    std::vector<GaussianReport> zs({z});
    PositionSensor<Filter::Target> s;
    for (double t = 0.0; t < 5; t += 1) {
        std::cout << "\n\nTime: " << t << std::endl;
        m = Eigen::Vector2d({t, t});
        z.reset();
        lmb.correct(zs, s, t);
        std::cout << "Targets: " << lmb.targettree.targets.size() << std::endl;
        for (auto& t : lmb.targettree.targets) {
            std::cout << "\n\tTarget " << t->id << " (" << t << ")" << std::endl;
            std::cout << "\t\tWeight: " << t->r << std::endl;
            std::cout << "\t\tMean: " << t->pdf.mean().transpose() << std::endl;
            std::cout << "\t\tPDF: " << t->pdf << std::endl;
            std::cout << "\t\tCov: " << std::endl << t->pdf.cov() << std::endl;
        }
    }
    EXPECT_EQ(lmb.targettree.targets.size(), 2);
}

TEST(LMBTests, OSPA) {
    using Filter = SILMB<GM<4>>;
    Filter lmb;
    GaussianReport::Measurement m(2); m.setZero();
    GaussianReport::Covariance P(4, 4); P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 3);
    std::vector<GaussianReport> zs({z});
    double c = 1;
    double p = 2;
    typename Filter::TargetStates truth1{{0, 0, 0, 0}};
    typename Filter::TargetStates truth2{{1, 0, 0, 0}};
    typename Filter::TargetStates truth3{{1, 0, 0, 0}, {0, 0, 0, 0}};
    typename Filter::TargetStates truth4{{0, 0, 0, 0}, {0, 0, 1, 0}};
    typename Filter::TargetStates truth5{};
    typename Filter::TargetStates truth6{{0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}};

    PositionSensor<Filter::Target> s;
    ASSERT_EQ(lmb.targettree.targets.size(), 0);
    EXPECT_FLOAT_EQ(lmb.ospa(truth1, c, p), c);
    EXPECT_FLOAT_EQ(lmb.ospa(truth2, c, p), c);
    EXPECT_FLOAT_EQ(lmb.ospa(truth6, c, p), c);
    EXPECT_FLOAT_EQ(lmb.ospa(truth1, c, p), lmb.ospa(truth4, c, p));

    lmb.correct(zs, s, 0);

    ASSERT_EQ(lmb.targettree.targets.size(), 1);
    EXPECT_FLOAT_EQ(lmb.ospa(truth1, c, p), 0);
    EXPECT_FLOAT_EQ(lmb.ospa(truth2, c, p), 1);
    EXPECT_FLOAT_EQ(lmb.ospa(truth3, c, p), lmb.ospa(truth4, c, p));
    EXPECT_FLOAT_EQ(lmb.ospa(truth2, c, p), lmb.ospa(truth5, c, p));
}

TEST(LMBTests, GOSPA) {
    using Filter = SILMB<GM<4>>;
    Filter lmb;
    GaussianReport::Measurement m(2); m.setZero();
    GaussianReport::Covariance P(4, 4); P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 3);
    std::vector<GaussianReport> zs({z});
    double c = 1;
    double p = 1;
    typename Filter::TargetStates truth1{{0, 0, 0, 0}};
    typename Filter::TargetStates truth2{{1, 0, 0, 0}};
    typename Filter::TargetStates truth3{{1, 0, 0, 0}, {0, 0, 0, 0}};
    typename Filter::TargetStates truth4{{0, 0, 0, 0}, {0, 0, 1, 0}};
    typename Filter::TargetStates truth5{};
    typename Filter::TargetStates truth6{{0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}};

    PositionSensor<Filter::Target> s;
    ASSERT_EQ(lmb.targettree.targets.size(), 0);
    EXPECT_FLOAT_EQ(lmb.gospa(truth1, c, p), c / 2);
    EXPECT_FLOAT_EQ(lmb.gospa(truth2, c, p), c / 2);
    EXPECT_FLOAT_EQ(lmb.gospa(truth6, c, p), 2 * c);
    EXPECT_LT(lmb.gospa(truth1, c, p), lmb.gospa(truth4, c, p));

    lmb.correct(zs, s, 0);

    ASSERT_EQ(lmb.targettree.targets.size(), 1);
    EXPECT_FLOAT_EQ(lmb.gospa(truth1, c, p), 0);
    EXPECT_FLOAT_EQ(lmb.gospa(truth2, c, p), 1);
    EXPECT_FLOAT_EQ(lmb.gospa(truth3, c, p), lmb.gospa(truth4, c, p));
    EXPECT_GT(lmb.gospa(truth2, c, p), lmb.gospa(truth5, c, p));
}
