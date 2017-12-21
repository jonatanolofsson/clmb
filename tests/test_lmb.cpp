#include <gtest/gtest.h>
#include <Eigen/Core>
#include "lmb.hpp"
#include "gm.hpp"
#include "sensors.hpp"
#include "models.hpp"
#include "params.hpp"

using namespace lmb;


TEST(LMBTests, ConstructLMB) {
    typedef SILMB<GM<4>> Filter;
    Params params;
    Filter lmb(&params);
    GaussianReport::Measurement m(2); m = Eigen::Vector2d({1, 1});
    GaussianReport::Covariance P(4, 4); P = Eigen::Matrix2d::Identity();
    GaussianReport z(m, P, 3);
    std::vector<GaussianReport> zs({z});
    PositionSensor<Filter::Target> s;
    lmb.correct(zs, s, 1);
    ASSERT_EQ(lmb.targets.targets.size(), 1);
    EXPECT_EQ(lmb.targets.targets[0]->id, 0);
    EXPECT_FLOAT_EQ(lmb.targets.targets[0]->r, params.rB_max);
    auto mean = lmb.targets.targets[0]->pdf.mean();
    EXPECT_FLOAT_EQ(mean[0], 1);
    EXPECT_FLOAT_EQ(mean[1], 1);
}

TEST(LMBTests, RunLMB) {
    typedef SILMB<GM<4>> Filter;
    Params params;
    Filter lmb(&params);
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
        std::cout << "Targets: " << lmb.targets.targets.size() << std::endl;
        for (auto& t : lmb.targets.targets) {
            std::cout << "\n\tTarget " << t->id << " (" << t << ")" << std::endl;
            std::cout << "\t\tWeight: " << t->r << std::endl;
            std::cout << "\t\tMean: " << t->pdf.mean().transpose() << std::endl;
            std::cout << "\t\tPDF: " << t->pdf << std::endl;
            std::cout << "\t\tCov: " << std::endl << t->pdf.cov() << std::endl;
        }
    }
    ASSERT_EQ(lmb.targets.targets.size(), 2);
}
