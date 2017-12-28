#include <gtest/gtest.h>
#include <Eigen/Core>
#include <gm.hpp>
#include <bbox.hpp>
#include <targettree.hpp>

using namespace lmb;

TEST(TargetTreeTests, InsertAndQuery) {
    typedef GM<4> PDF;
    TargetTree<PDF> targets;
    Params params;
    targets.new_target(1.0, PDF(&params, {0, 0, 0, 0}, PDF::Covariance::Identity()), 0);
    targets.new_target(0.5, PDF(&params, {1, 1, 1, 1}, PDF::Covariance::Identity()), 0);
    ASSERT_EQ(targets.targets.size(), 2);
    TargetTree<PDF>::Targets res = targets.query(AABBox(-1, -1, 1, 1));
    EXPECT_EQ(res.size(), 2);
    EXPECT_NE(res[0], res[1]);
    EXPECT_EQ(res[0], targets.targets[0]);
    EXPECT_EQ(res[1], targets.targets[1]);
    EXPECT_FLOAT_EQ(res[0]->r, 1.0);
    EXPECT_FLOAT_EQ(res[1]->r, 0.5);
    res = targets.query(AABBox(-30000, -30000, 30000, 30000));
    EXPECT_EQ(res.size(), 2);
    EXPECT_NE(res[0], res[1]);
    EXPECT_EQ(res[0], targets.targets[0]);
    EXPECT_EQ(res[1], targets.targets[1]);
    EXPECT_FLOAT_EQ(res[0]->r, 1.0);
    EXPECT_FLOAT_EQ(res[1]->r, 0.5);
}

TEST(TargetTreeTests, Query2) {
    typedef GM<4> PDF;
    TargetTree<PDF> targets;
    Params params;
    targets.new_target(1.0, PDF(&params, {0, 0, 0, 0}, PDF::Covariance::Identity()), 0);
    TargetTree<PDF>::Targets res = targets.query(AABBox(-0.1, -0.1, 0.1, 0.1));
    EXPECT_EQ(res.size(), 1);
}
