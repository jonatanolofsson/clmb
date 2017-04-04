#include "hypgen.hpp"

namespace lmb {
    Murty::Murty(const CostMatrix C) {
        size = C.cols();
        state.C = C;
    }

    void Murty::draw(Assignment&, double&) {
    }
}
