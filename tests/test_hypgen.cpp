#include <gtest/gtest.h>
#include <Eigen/Core>
#include <murty.hpp>
#include <limits>

using namespace lap;

lap::CostMatrix MURTY_COST = (lap::CostMatrix(10, 10) <<
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


lap::CostMatrix MURTY_COST2 = (lap::CostMatrix(3, 3) <<
              0, 0, 46,
              0, 44, 13,
              3, 14, 0).finished();

//double linf = std::numeric_limits<double>::infinity();
double linf = lap::inf;
lap::CostMatrix MURTY_HARD = (lap::CostMatrix(4, 10) <<
  -1.92591,   -2.88444,       linf,      30000,      30000,      30000,    17.8891,      30000,      30000,      30000,
  -1.92591,   -2.88444,      30000,       linf,      30000,      30000,      30000,    17.8891,      30000,      30000,
-0.0512712, -0.0512712,      30000,      30000,       linf,      30000,      30000,      30000,   0.287682,      30000,
-0.0512712, -0.0512712,      30000,      30000,      30000,       linf,      30000,      30000,      30000,      30000
).finished();


TEST(LAPTests, SingleLAP) {
    auto C = MURTY_COST;
    Assignment res(C.rows());
    Dual u(C.rows());
    Dual v(C.cols());
    lap::lap(C, res, u, v);
    ASSERT_EQ(res, (Assignment(10) << 8, 6, 2, 7, 5, 3, 9, 0, 4, 1).finished());
}

TEST(LAPTests, SmallSingle) {
    auto C = MURTY_COST2;
    Assignment res(C.rows());
    Dual u(C.rows());
    Dual v(C.cols());
    lap::lap(C, res, u, v);
    ASSERT_EQ(res, (Assignment(3) << 1, 0, 2).finished());
}

TEST(LAPTests, SingleLAP_Asym) {
    auto C = MURTY_COST.block<5, 10>(0, 0);
    Assignment res(C.rows());
    Dual u(C.rows());
    Dual v(C.cols());
    lap::lap(C, res, u, v);
    ASSERT_EQ(res, (Assignment(5) << 8, 6, 2, 1, 0).finished());
}

TEST(LAPTests, SingleLAP_Hard) {
    auto C = MURTY_HARD;
    C.array() -= C.minCoeff();
    Assignment res(C.rows());
    Dual u(C.rows());
    Dual v(C.cols());
    lap::lap(C, res, u, v);
    ASSERT_EQ(res, (Assignment(4) << 6, 1, 8, 0).finished());
    //ASSERT_EQ(res, (Assignment(4) << 1, 7, 8, 0).finished());
}

TEST(LAPTests, SingleMurtyState) {
    auto C = MURTY_COST;
    MurtyState s(C);
    s.solve();
    s.v.setZero();
    auto res = s.solution;
    auto cost = s.cost;
    double cumsum = 0;
    for (int i = 0; i < res.size(); ++i) { cumsum += C(i, res[i]); }
    ASSERT_DOUBLE_EQ(cumsum, cost);
    ASSERT_EQ(s.solution, (Assignment(10) << 8, 6, 2, 7, 5, 3, 9, 0, 4, 1).finished());
}

TEST(MurtyTests, MiniMurty) {
    auto C = Eigen::Matrix<double, 1, 1>(0);
    Murty m(C);
    Assignment res;
    double cost;
    m.draw(res, cost);
    double cumsum = 0;
    for (int i = 0; i < res.size(); ++i) { cumsum += C(i, res[i]); }
    ASSERT_DOUBLE_EQ(cumsum, cost);
    ASSERT_EQ(res, (Assignment(1) << 0).finished());
}

TEST(MurtyTests, HardSingle) {
    auto C = MURTY_HARD;
    Murty m(C);
    Assignment res;
    double cost;
    m.draw(res, cost);
    double cumsum = 0;
    for (int i = 0; i < res.size(); ++i) { cumsum += C(i, res[i]); }
    ASSERT_DOUBLE_EQ(cumsum, cost);
    ASSERT_EQ(res, (Assignment(4) << 6, 1, 8, 0).finished());
}

TEST(MurtyTests, Hard) {
    auto C = MURTY_HARD;
    Murty m(C);
    Assignment res;
    double cost, last_cost = 0;
    unsigned n = 0;
    while (m.draw(res, cost)) {
        ++n;
        double cumsum = 0;
        for (int i = 0; i < res.size(); ++i) { cumsum += C(i, res[i]); }
        ASSERT_DOUBLE_EQ(cumsum, cost);
        ASSERT_GE(cost - last_cost, -1e-8);
        last_cost = cost;
    }
    ASSERT_EQ(n, 3333);
}

TEST(MurtyTests, SmallMurty) {
    lap::CostMatrix C = (lap::CostMatrix(1, 3) << 2.14843, lap::inf + 2, 1.20397).finished();
    Murty m(C);
    Assignment res;
    double cost, last_cost = 0;
    unsigned n = 0;
    while (m.draw(res, cost)) {
        ++n;
        double cumsum = 0;
        for (int i = 0; i < res.size(); ++i) { cumsum += C(i, res[i]); }
        ASSERT_DOUBLE_EQ(cumsum, cost);
        ASSERT_GE(cost, last_cost);
        last_cost = cost;
    }
    ASSERT_EQ(n, 2);
}

TEST(MurtyTests, SingleMurty) {
    auto C = MURTY_COST;
    Murty m(C);
    Assignment res;
    double cost;
    m.draw(res, cost);
    double cumsum = 0;
    for (int i = 0; i < res.size(); ++i) { cumsum += C(i, res[i]); }
    ASSERT_DOUBLE_EQ(cumsum, cost);
    ASSERT_EQ(res, (Assignment(10) << 8, 6, 2, 7, 5, 3, 9, 0, 4, 1).finished());
}

TEST(MurtyTests, FullMurty) {
    auto C = MURTY_COST.block<7, 7>(0, 0);
    Murty m(C);
    Assignment res;
    double cost, last_cost = 0;
    unsigned n = 0;
    while (m.draw(res, cost)) {
        ++n;
        double cumsum = 0;
        //std::cout << "N: " << n << ": ";
        for (int i = 0; i < res.size(); ++i) { cumsum += C(i, res[i]); }
        ASSERT_DOUBLE_EQ(cumsum, cost);
        ASSERT_GE(cost, last_cost);
        last_cost = cost;
    }
    //ASSERT_EQ(n, 720);  // 6
    ASSERT_EQ(n, 5040);  // 7
    //ASSERT_EQ(n, 40320);  // 8
    //ASSERT_EQ(n, 362880);  // 9
    //ASSERT_EQ(n, 3628800);  // 10
}

TEST(MurtyTests, RectMurty) {
    auto C = MURTY_COST.block<2, 10>(0, 0);
    Murty m(C);
    Assignment res;
    double cost, last_cost = 0;
    unsigned n = 0;
    while (m.draw(res, cost)) {
        ++n;
        double cumsum = 0;
        for (int i = 0; i < res.size(); ++i) { cumsum += C(i, res[i]); }
        ASSERT_DOUBLE_EQ(cumsum, cost);
        ASSERT_GE(cost, last_cost);
        last_cost = cost;
    }
    ASSERT_EQ(n, 90);
}
