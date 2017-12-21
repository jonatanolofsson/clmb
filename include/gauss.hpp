#pragma once
#include <Eigen/Core>

namespace lmb {
    template<int S>
    struct alignas(16) GaussianComponent {
        static const int STATES = S;
        typedef GaussianComponent<S> Self;
        typedef Eigen::Matrix<double, STATES, 1> State;
        typedef Eigen::Matrix<double, STATES, STATES> Covariance;
        State m;
        Covariance P;
        double w;

        GaussianComponent(double w_, const State& m_, const Covariance& P_)
        : m(m_),
          P(P_),
          w(w_)
        {}

        bool operator<(const GaussianComponent& b) const {
            return w < b.w;
        }

        void repr(std::ostream& os) const {
            os << "{\"type\":\"G\",\"w\":" << w << ",\"m\":" << m.format(eigenformat) << ",\"P\":" << P.format(eigenformat) << "}";
        }
    };

    template<int S>
    auto& operator<<(std::ostream& os, const GaussianComponent<S> c) {
        c.repr(os);
        return os;
    }
}
