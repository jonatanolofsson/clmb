#pragma once
#include <Eigen/Core>
#include "bbox.hpp"
#include "statistics.hpp"
#include "omp.hpp"

namespace lmb {
    template<int S>
    struct alignas(16) Gaussian {
        static const int STATES = S;
        typedef Gaussian<S> Self;
        typedef Eigen::Matrix<double, STATES, 1> State;
        typedef Eigen::Matrix<double, STATES, STATES> Covariance;
        State m;
        Covariance P;
        double w;

        Gaussian(double w_, const State& m_, const Covariance& P_)
        : m(m_),
          P(P_),
          w(w_)
        {}

        template<typename Report, typename Sensor>
        double correct(const Report& z, const Sensor& s) {
            auto dz = (z.z - s.measurement(m)).eval();
            auto Dinv = (s.H * P * s.H.transpose() + z.R).inverse().eval();
            auto K = P * s.H.transpose() * Dinv;
            w *= s.pD(*this) * z.likelihood(dz, Dinv) / z.kappa;
            m += K * dz;
            P -= K * s.H * P;

            return w;
        }

        bool operator<(const Gaussian& b) const {
            return w < b.w;
        }

        State mean() {
            return m;
        }

        Covariance cov() {
            return P;
        }

        template<typename RES>
        void sample(RES& res) const {
            nrand(res, m, P);
        }

        double overlap(const BBox& bbox) const {
            static const int N = 10000;
            Eigen::Matrix<double, S, Eigen::Dynamic> samples(S, N);
            sample(samples);
            int nwithin = 0;
            PARFOR
            for(int n = 0; n < N; ++n) {
                if(bbox.within(samples.col(n))) {
                    ++nwithin;
                }
            }
            return ((double)nwithin) / N;
        }

        void repr(std::ostream& os) const {
            os << "{\"type\":\"G\",\"w\":" << w << ",\"m\":" << m.format(eigenformat) << ",\"P\":" << P.format(eigenformat) << "}";
        }
    };

    template<int S>
    auto& operator<<(std::ostream& os, const Gaussian<S> c) {
        c.repr(os);
        return os;
    }
}
