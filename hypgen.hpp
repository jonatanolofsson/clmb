#pragma once

#include <Eigen/Core>

namespace lmb {
    typedef Eigen::RowVectorXi Assignment;
    typedef Eigen::MatrixXd CostMatrix;

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

    void lap(const CostMatrix&, Assignment&, double&);
    void lapjv(const CostMatrix&, Assignment&, double&);
}
