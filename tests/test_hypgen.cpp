#include <gtest/gtest.h>
#include <Eigen/Core>
#include <murty.hpp>
#include <limits>

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

double linf = std::numeric_limits<double>::infinity();
Eigen::MatrixXd MURTY_HARD = (Eigen::MatrixXd(4, 10) <<
  -1.92591,   -2.88444,       linf,      30000,      30000,      30000,    17.8891,      30000,      30000,      30000,
  -1.92591,   -2.88444,      30000,       linf,      30000,      30000,      30000,    17.8891,      30000,      30000,
-0.0512712, -0.0512712,      30000,      30000,       linf,      30000,      30000,      30000,   0.287682,      30000,
-0.0512712, -0.0512712,      30000,      30000,      30000,       linf,      30000,      30000,      30000,      30000
).finished();


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

TEST(LAPTests, SingleLAP_Asym) {
    Assignment res(MURTY_COST_ASYM.rows());
    Slack u(MURTY_COST_ASYM.rows());
    Slack v(MURTY_COST_ASYM.cols());
    lap::lap(MURTY_COST_ASYM, res, u, v);
    ASSERT_EQ(res, (Assignment(5) << 8, 6, 2, 1, 0).finished());
}

TEST(LAPTests, SingleMurtyState) {
    MurtyState s(MURTY_COST);
    s.solve();
    s.v.setZero();
    ASSERT_EQ(s.solution, (Assignment(10) << 8, 6, 2, 7, 5, 3, 9, 0, 4, 1).finished());
}

TEST(MurtyTests, MiniMurty) {
    Murty m(Eigen::Matrix<double, 1, 1>(0));
    Assignment res;
    double cost;
    m.draw(res, cost);
    ASSERT_EQ(res, (Assignment(1) << 0).finished());
}

TEST(MurtyTests, HardSingle) {
    Murty m(MURTY_HARD);
    Assignment res;
    double cost;
    m.draw(res, cost);
    ASSERT_EQ(res, (Assignment(4) << 6, 1, 8, 0).finished());
}

TEST(MurtyTests, SmallMurty) {
    Eigen::MatrixXd C = (Eigen::MatrixXd(1, 3) << 2.14843, inf, 1.20397).finished();
    Murty m(C);
    Assignment res;
    double cost;
    unsigned n = 0;
    while(m.draw(res, cost)) { ++n; }
    ASSERT_EQ(n, 2);
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
