#pragma once

#include "lap.hpp"

namespace lmb {
    using namespace lap;
    class Murty {
        public:
            Murty(const CostMatrix);

            void draw(Assignment&, double&);

        private:
            struct State {
                CostMatrix C;
            };
            State state;
            int size;
    };
}
