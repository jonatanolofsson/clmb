#pragma once
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <vector>
#include "constants.hpp"
#include "omp.hpp"
#include "params.hpp"
#include "sensors.hpp"
#include "statistics.hpp"

#include <iostream>

namespace lmb {
    static const double pfw_lim = 0.05; // FIXME

    template<unsigned S, unsigned N = 200000>
    struct PF {
        static const unsigned STATES = S;
        static const unsigned PARTICLES = N;
        typedef PF<S, PARTICLES> Self;
        typedef Eigen::Matrix<double, STATES, 1> State;
        typedef Eigen::Matrix<double, STATES, STATES> Covariance;
        typedef Eigen::Array<double, STATES, Eigen::Dynamic> States;
        typedef Eigen::Array<double, 1, Eigen::Dynamic> Weights;
        Params* params;

        AABBox aabbox;
        double eta;
        Weights w;
        States x;
        bool valid = false;
        std::size_t permutation[N];

        PF()
        : w(1, N),
          x(S, N)
        {
            std::iota(std::begin(permutation), std::end(permutation), 0);
        }

        PF(Params* params_)
        : params(params_),
          w(1, N),
          x(S, N)
        {
            std::iota(std::begin(permutation), std::end(permutation), 0);
        }

        PF(Params* params_, const State& mean, const Covariance& covariance)
        : params(params_),
          w(1, N),
          x(S, N)
        {
            std::iota(std::begin(permutation), std::end(permutation), 0);
            nrand(x, mean, covariance);
            w.setConstant(1.0 / N);
        }

        template<typename FT, typename QT>
        void linear_update(const FT& F, const QT&) {
            PARFOR
            for (unsigned i = 0; i < x.cols(); ++i) {
                x.col(i).matrix() = F * x.col(i).matrix();
            }
        }

        template<typename Measurement, typename Sensor>
        double correct(const Measurement& z, const Sensor& s) {
            auto Dinv = z.R.inverse().eval();
            PARFOR
            for(unsigned i = 0; i < N; ++i) {
                auto dz = (z.z - s.measurement(x.col(i).matrix())).eval();
                w[i] *= s.pD(x.col(i)) * z.likelihood(dz, Dinv) / z.kappa;
            }
            normalize();
            return eta;
        }

        void normalize() {
            eta = w.sum();
            if (eta > 1e-8) {
                w /= eta;
            } else {
                invalidate();
            }
            update_bbox();
        }

        void update_bbox(const double lim=0.01) {  // FIXME: Change to nstd
            if (!valid) { return; }

            aabbox.min[0] = std::numeric_limits<double>::infinity();
            aabbox.min[1] = std::numeric_limits<double>::infinity();
            aabbox.max[0] = -std::numeric_limits<double>::infinity();
            aabbox.max[1] = -std::numeric_limits<double>::infinity();

            for(unsigned i = 0; i < N; ++i) {
                if (w[i] < lim) { continue; }
                aabbox.min[0] = std::min(aabbox.min[0], x(0, i));
                aabbox.min[1] = std::min(aabbox.min[1], x(1, i));
                aabbox.max[0] = std::max(aabbox.max[0], x(0, i));
                aabbox.max[1] = std::max(aabbox.max[1], x(1, i));
            }
        }

        double neff() const {
            return 1 / std::accumulate(std::begin(w), std::end(w), 0.0,
                                       [&](auto k, auto l) { return k + l * l; });
        }

        void resample() {
            if(!valid) { return; }
            auto cdf = std::partial_sum(std::begin(w), std::end(w));
            auto w0 = 1.0 / N;
            double u0 = urand() * w0;
            unsigned j = 0;

            States x2;
            PARFOR
            for(unsigned i = 0; i < N; ++i) {
                double u = u0 + i * w0;
                while(cdf[j] < u) { ++j; }
                x2.col(i) = x.col(j);
            }
            x = x2;
            w.setConstant(w0);
        }

        void invalidate() {
            valid = false;
            eta = 0;
        }

        template<typename Sensor>
        double missed(const Sensor& s) {
            PARFOR
            for(unsigned i = 0; i < N; ++i) {
                w[i] *= s.pD(x.col(i));
            }
            normalize();
            return 1 - eta;
        }

        void repr(std::ostream& os) const {
            os << "{\"type\":\"PF\"";
            os << ",\"w\":" << w.format(eigenformat);
            os << ",\"x\":" << x.format(eigenformat);
            os << "}";
        }

        void prepare_sort() {
            std::sort(std::begin(permutation), std::end(permutation),
                      [&](std::size_t i, std::size_t j){ return w[i] > w[j]; });
        }

        double wsorted(const unsigned i) {
            return w[permutation[i]];
        }

        State msorted(const unsigned i) {
            return x.col(permutation[i]);
        }

        template<typename W>
        static void join(Self& pdf, const W& pdf_weights, std::vector<Self>& pdfs) {
            std::priority_queue<std::tuple<double, unsigned, unsigned>> q;

            for(unsigned j = 0; j < pdf_weights.cols(); ++j) {
                pdfs[j].prepare_sort();
                q.emplace(pdf_weights[j] * pdfs[j].wsorted(0), j, 0);
            }

            for(unsigned i = 0; i < N; ++i) {
                auto& p = q.top();
                double nw = std::get<0>(p);
                unsigned j = std::get<1>(p);
                unsigned n = std::get<2>(p);
                pdf.w[i] = nw;
                pdf.x.col(i) = pdfs[j].msorted(n);
                q.pop();
                ++n;
                if (n < N) {
                    q.emplace(pdf_weights[j] * pdfs[j].wsorted(n), j, n);
                }
            }
            pdf.normalize();
        }

        State mean() const {
            return (x.array().rowwise() * w).rowwise().sum();
        }

        Covariance cov(const State& m) const {
            // See https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance
            auto d = (x.colwise() - m.array());
            return (d.rowwise() * w).matrix() * d.matrix().transpose() / (1 - (w * w).sum());
        }

        Covariance cov() const {
            return cov(mean());
        }
    };

    template<unsigned STATES, unsigned PARTICLES = 200>
    auto& operator<<(std::ostream& os, const PF<STATES, PARTICLES>& t) {
        t.repr(os);
        return os;
    }
}
