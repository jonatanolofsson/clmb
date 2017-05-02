#include <gtest/gtest.h>
#include <Eigen/Core>
#include "connectedcomponents.hpp"

using namespace lmb;

TEST(CCTests, InsertAndQuery) {
    int ints[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ConnectedComponents<int> cc;
    for (unsigned i = 0; i < 10; ++i) {
        cc.init(ints + i);
    }
    cc.connect(ints + 0, {ints + 1, ints + 2, ints + 3, ints + 4});
    cc.connect(ints + 4, {ints + 5, ints + 6, ints + 3, ints + 4});
    cc.connect(ints + 7, {ints + 8, ints + 9});
    unsigned n = 0;
    std::vector<int*> res;
    int expn[2] = {7, 3};
    while(cc.get_component(res)) {
        ASSERT_LT(n, 2);
        EXPECT_EQ(res.size(), expn[n]);
        ++n;
    }
    EXPECT_EQ(n, 2);
}
