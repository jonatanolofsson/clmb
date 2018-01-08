#pragma once
#include <Eigen/Core>
#include "bbox.hpp"
#include "statistics.hpp"
#include "omp.hpp"

namespace lmb {
    template<int S>
    struct alignas(16) Gaussian_ {
        static const int STATES = S;
        using Self = Gaussian_<S>;
        using State = Eigen::Matrix<double, STATES, 1>;
        using Covariance = Eigen::Matrix<double, STATES, STATES>;
        State m;
        Covariance P;
        double w;

        Gaussian_(const State& m_, const Covariance& P_, double w_=1.0)
        : m(m_),
          P(P_),
          w(w_)
        {}

        template<typename Sensor>
        double correct(const typename Sensor::Report& z, const Sensor& s) {
            auto dz = (z.z - s.measurement(m)).eval();
            auto Dinv = (s.H * P * s.H.transpose() + z.R).inverse().eval();
            auto K = P * s.H.transpose() * Dinv;
            w *= s.pD(*this) * z.likelihood(dz, Dinv) / z.kappa;
            m += K * dz;
            P -= K * s.H * P;

            return w;
        }

        bool operator<(const Gaussian_& b) const {
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
            static const int N = 10000;  // FIXME
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
    auto& operator<<(std::ostream& os, const Gaussian_<S> c) {
        c.repr(os);
        return os;
    }
}
