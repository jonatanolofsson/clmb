#include <random>
#include <Eigen/Eigenvalues>
#include "statistics.hpp"

namespace lmb {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    static std::uniform_real_distribution<> udist;
    static std::normal_distribution<> ndist;

    double urand() {
        return udist(gen);
    }

    double nrand() {
        return ndist(gen);
    }
}
