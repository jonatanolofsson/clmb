#include <gtest/gtest.h>
#include <Eigen/Core>
#include "bbox.hpp"
#include "constants.hpp"

using namespace lmb;
typedef Eigen::Matrix<double, 2, 4> Corners;

TEST(BBoxTests, DefaultInit) {
    BBox bbox;
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(bbox.corners(i, j), ((i == 0) ? (j >= 2) : ((j + 1) % 4) < 2) ? inf : -inf);
        }
    }
}

TEST(BBoxTests, Init) {
    Corners corners;  corners << 0, 0, 1, 1,
                                 1, 0, 0, 1;
    BBox bbox(corners);
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(bbox.corners(i, j), corners(i, j));
        }
    }
}

TEST(BBoxTests, Within) {
    Corners corners;  corners << 0, 0, 1, 1,
                                 1, 0, 0, 1;
    BBox bbox(corners);
    Eigen::Vector2d x; x << 0.5, 0.5;
    EXPECT_TRUE(bbox.within(x));
}

TEST(BBoxTests, Without) {
    Corners corners;  corners << 0, 0, 1, 1,
                                 1, 0, 0, 1;
    BBox bbox(corners);
    Eigen::Vector2d x; x << 1.5, 1.5;
    EXPECT_FALSE(bbox.within(x));
}

TEST(BBoxTests, Intersect) {
    Corners corners; corners << 0, 0, 1, 1,
                                1, 0, 0, 1;
    Corners corners2; corners2 << 0.5, 0.5, 1.5, 1.5,
                                  1.5, 0.5, 0.5, 1.5;
    BBox bbox(corners);
    BBox bbox2(corners2);
    EXPECT_TRUE(bbox.intersects(bbox2));
}

TEST(BBoxTests, Disjoint) {
    Corners corners; corners << 0, 0, 1, 1,
                                1, 0, 0, 1;
    Corners corners2; corners2 << 2.5, 1.5, 2.5, 2.5,
                                  2.5, 1.5, 1.5, 2.5;
    BBox bbox(corners);
    BBox bbox2(corners2);
    EXPECT_FALSE(bbox.intersects(bbox2));
}
