// Copyright 2018 Jonatan Olofsson
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <cf.hpp>


TEST(CFTests, nedrot0) {
    Eigen::Vector3d origin; origin << 0, 0, 0;
    Eigen::Matrix3d ref;
    ref << 0,  0, 1,
           0,  1, 0,
           -1, 0, 0;
    auto R = cf::nedrot(origin);
    EXPECT_DOUBLE_EQ((R - ref).norm(), 0);
}

TEST(CFTests, nedrot_np) {
    Eigen::Vector3d origin; origin << 90, 0, 0;
    Eigen::Matrix3d ref;
    ref << -1, 0, 0,
           0,  1, 0,
           0,  0, -1;
    auto R = cf::nedrot(origin);
    EXPECT_NEAR((R - ref).norm(), 0, 1e-16);
}

TEST(CFTests, ecefned) {
    Eigen::Vector3d lla; lla << 58.398566, 15.577402, 100;
    auto ecef = cf::lla2ecef(lla);
    Eigen::Vector3d origin; origin << 58.3887657, 15.6965082, 80;
    auto ned = cf::ecef2ned(ecef, origin);
    auto res = cf::ned2ecef(ned, origin);
    EXPECT_DOUBLE_EQ((res - ecef).norm(), 0);
}

TEST(CFTests, llaned) {
    Eigen::Vector3d lla; lla << 58.398566, 15.577402, 100;
    Eigen::Vector3d origin; origin << 58.3887657, 15.6965082, 80;
    auto ned = cf::lla2ned(lla, origin);
    auto res = cf::ned2lla(ned, origin);
    EXPECT_NEAR((res - lla).norm(), 0, 1e-9);
}

TEST(CFTests, llaecef) {
    Eigen::Vector3d lla; lla << 58.398566, 15.577402, 100;
    auto ecef = cf::lla2ecef(lla);
    auto res = cf::ecef2lla(ecef);
    EXPECT_NEAR((res - lla).norm(), 0, 1e-9);
}
