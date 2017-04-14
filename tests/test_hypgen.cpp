#include <gtest/gtest.h>
#include <Eigen/Core>
#include <murty.hpp>

using namespace lmb;

Eigen::MatrixXd MURTY_COST = (Eigen::MatrixXd(10, 10) <<
              7, 51, 52, 87, 38, 60, 74, 66, 0, 20,
              50, 12, 0, 64, 8, 53, 0, 46, 76, 42,
              27, 77, 0, 18, 22, 48, 44, 13, 0, 57,
              62, 0, 3, 8, 5, 6, 14, 0, 26, 39,
              0, 97, 0, 5, 13, 0, 41, 31, 62, 48,
              79, 68, 0, 0, 15, 12, 17, 47, 35, 43,
              76, 99, 48, 27, 34, 0, 0, 0, 28, 0,
              0, 20, 9, 27, 46, 15, 84, 19, 3, 24,
              56, 10, 45, 39, 0, 93, 67, 79, 19, 38,
              27, 0, 39, 53, 46, 24, 69, 46, 23, 1).finished();


Eigen::MatrixXd MURTY_COST2 = (Eigen::MatrixXd(3, 3) <<
              0, 0, 46,
              0, 44, 13,
              3, 14, 0).finished();


Eigen::MatrixXd MURTY_HARD = (Eigen::MatrixXd(4, 4) <<
    7,    52,    87,    38,
   27, 30000,    18, 30000,
   62, 30000,     8,     5,
   79, 30000, 30000, 30000).finished();


Eigen::MatrixXd MURTY_COST_ASYM = (Eigen::MatrixXd(5, 10) <<
              7, 51, 52, 87, 38, 60, 74, 66, 0, 20,
              50, 12, 0, 64, 8, 53, 0, 46, 76, 42,
              27, 77, 0, 18, 22, 48, 44, 13, 0, 57,
              62, 0, 3, 8, 5, 6, 14, 0, 26, 39,
              0, 97, 0, 5, 13, 0, 41, 31, 62, 48).finished();

TEST(LAPTests, SingleLAP) {
    Assignment res(MURTY_COST.rows());
    Slack u(MURTY_COST.rows());
    Slack v(MURTY_COST.cols());
    lap::lap(MURTY_COST, res, u, v);
    ASSERT_EQ(res, (Assignment(10) << 8, 6, 2, 7, 5, 3, 9, 0, 4, 1).finished());
}

TEST(LAPTests, SmallSingle) {
    Assignment res(MURTY_COST2.rows());
    Slack u(MURTY_COST2.rows());
    Slack v(MURTY_COST2.cols());
    lap::lap(MURTY_COST2, res, u, v);
    ASSERT_EQ(res, (Assignment(3) << 1, 0, 2).finished());
}

TEST(LAPTests, HardSingle) {
    Assignment res(MURTY_HARD.rows());
    Slack u(MURTY_HARD.rows());
    Slack v(MURTY_HARD.cols());
    lap::lap(MURTY_HARD, res, u, v);
    ASSERT_EQ(res, (Assignment(4) << 1, 2, 3, 0).finished());
}

TEST(LAPTests, SingleLAP_Asym) {
    Assignment res(MURTY_COST_ASYM.rows());
    Slack u(MURTY_COST_ASYM.rows());
    Slack v(MURTY_COST_ASYM.cols());
    lap::lap(MURTY_COST_ASYM, res, u, v);
    ASSERT_EQ(res, (Assignment(5) << 8, 6, 2, 1, 0).finished());
}

TEST(LAPTests, SingleState) {
    State s(MURTY_COST);
    s.solve();
    s.v.setZero();
    ASSERT_EQ(s.solution, (Assignment(10) << 8, 6, 2, 7, 5, 3, 9, 0, 4, 1).finished());
}

TEST(MurtyTests, SingleMurty) {
    Murty m(MURTY_COST);
    Assignment res;
    double cost;
    m.draw(res, cost);
    ASSERT_EQ(res, (Assignment(10) << 8, 6, 2, 7, 5, 3, 9, 0, 4, 1).finished());
}

TEST(MurtyTests, FullMurty) {
    Murty m(MURTY_COST.block<6, 6>(0, 0));
    Assignment res;
    double cost;
    unsigned n = 0;
    while(m.draw(res, cost)) { ++n; }
    ASSERT_EQ(n, 720);
}
