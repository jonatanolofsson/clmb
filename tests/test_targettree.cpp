// Copyright 2018 Jonatan Olofsson
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <gm.hpp>
#include <bbox.hpp>
#include <targettree.hpp>

using namespace lmb;

TEST(TargetTreeTests, InsertAndQuery) {
    using PDF = GM<4>;
    using TargetTree = TargetTree_<PDF>;
    Params params;
    TargetTree targettree(&params);
    targettree.new_target(1.0, PDF(&params, {0, 0, 0, 0}, PDF::Covariance::Identity()));  // NOLINT
    targettree.new_target(0.5, PDF(&params, {1, 1, 1, 1}, PDF::Covariance::Identity()));  // NOLINT
    ASSERT_EQ(targettree.targets.size(), 2);
    TargetTree::Targets res = targettree.query(AABBox(-1, -1, 1, 1));
    ASSERT_EQ(res.size(), 2);
    EXPECT_NE(res[0], res[1]);
    EXPECT_EQ(res[0], targettree.targets[0]);
    EXPECT_EQ(res[1], targettree.targets[1]);
    EXPECT_FLOAT_EQ(res[0]->r, 1.0);
    EXPECT_FLOAT_EQ(res[1]->r, 0.5);
    res = targettree.query(AABBox(-30000, -30000, 30000, 30000));
    EXPECT_EQ(res.size(), 2);
    EXPECT_NE(res[0], res[1]);
    EXPECT_EQ(res[0], targettree.targets[0]);
    EXPECT_EQ(res[1], targettree.targets[1]);
    EXPECT_FLOAT_EQ(res[0]->r, 1.0);
    EXPECT_FLOAT_EQ(res[1]->r, 0.5);
}

TEST(TargetTreeTests, Query2) {
    using PDF = GM<4>;
    using TargetTree = TargetTree_<PDF>;
    Params params;
    TargetTree targettree(&params);
    targettree.new_target(1.0, PDF(&params, {0, 0, 0, 0}, PDF::Covariance::Identity()));  // NOLINT
    TargetTree::Targets res = targettree.query(AABBox(-0.1, -0.1, 0.1, 0.1));
    EXPECT_EQ(res.size(), 1);
}
