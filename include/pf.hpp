// Copyright 2018 Jonatan Olofsson
#pragma once
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <tuple>
#include <vector>
#include "bbox.hpp"
#include "cf.hpp"
#include "constants.hpp"
#include "omp.hpp"
#include "params.hpp"
#include "sensors.hpp"
#include "statistics.hpp"


namespace lmb {

template<typename T>
struct Particle {
    const T& p;

    explicit Particle(const T& p_) : p(p_) {}

    double overlap(const BBox& bbox) const {
        return static_cast<double>(bbox.within(p));
    }
};

template<unsigned S, unsigned N = 200000>
struct PF {
    static const unsigned STATES = S;
    static const unsigned PARTICLES = N;
    using Self = PF<S, PARTICLES>;
    using State = Eigen::Matrix<double, STATES, 1>;
    using Covariance = Eigen::Matrix<double, STATES, STATES>;
    using States = Eigen::Array<double, STATES, Eigen::Dynamic>;
    using Weights = Eigen::Array<double, 1, Eigen::Dynamic>;
    Params* params;

    BBox bbox;
    AABBox aabbox;
    AABBox _llaabbox;
    double eta;
    Weights w;
    States x;
    bool valid = false;
    std::size_t permutation[N];
    cf::LL origin;

    PF()
    : w(1, N),
      x(S, N) {
        std::iota(std::begin(permutation), std::end(permutation), 0);
        transform_to_local(pos());
        transform_to_global();
    }

    explicit PF(Params* params_)
    : params(params_),
      w(1, N),
      x(S, N) {
        std::iota(std::begin(permutation), std::end(permutation), 0);
        transform_to_local(pos());
        transform_to_global();
    }

    PF(Params* params_, const State& mean, const Covariance& covariance)
    : params(params_),
      w(1, N),
      x(S, N) {
        std::iota(std::begin(permutation), std::end(permutation), 0);
        nrand(x, mean, covariance);
        w.setConstant(1.0 / N);
        transform_to_local(pos());
        transform_to_global();
    }

    template<typename FT, typename QT>
    void linear_update(const FT& F, const QT&) {
        PARFOR
        for (unsigned i = 0; i < x.cols(); ++i) {
            x.col(i).matrix() = F * x.col(i).matrix();
        }
    }

    template<typename Sensor>
    double correct(const typename Sensor::Report& z, const Sensor& s) {
        auto Dinv = z.cov().inverse().eval();
        PARFOR
        for (unsigned i = 0; i < N; ++i) {
            auto dz = (z.mean() - s.measurement(x.col(i).matrix())).eval();
            w[i] *= s.pD(Particle(x.col(i)), origin) * z.likelihood(dz, Dinv) / z.kappa;
        }
        normalize();
        return eta;
    }

    template<typename Sensor>
    double missed(const Sensor& s) {
        PARFOR
        for (unsigned i = 0; i < N; ++i) {
            w[i] *= s.pD(Particle(x.col(i)), origin);
        }
        normalize();
        return 1 - eta;
    }

    void normalize() {
        eta = w.sum();
        if (eta > 1e-8) {
            w /= eta;
        } else {
            invalidate();
        }
    }

    void update_local_bbox() {
        if (!valid) { return; }
        bbox.from_gaussian(pos(), poscov(), params->nstd);
        aabbox = bbox.aabbox();
    }

    bool intersects(const BBox& fov) {
        return bbox.intersects(fov);
    }

    bool intersects(const AABBox& fov) {
        return _llaabbox.intersects(fov);
    }

    double neff() const {
        return 1 / std::accumulate(
            std::begin(w), std::end(w), 0.0,
            [&](auto k, auto l) { return k + l * l; });
    }

    void resample() {
        if (!valid) { return; }
        std::vector<double> cdf(w.size());
        std::partial_sum(std::begin(w), std::end(w), std::begin(cdf));
        auto w0 = 1.0 / N;
        double u0 = urand() * w0;
        unsigned j = 0;

        States x2; x2.resize(STATES, N);
        PARFOR
        for (unsigned i = 0; i < N; ++i) {
            double u = u0 + i * w0;
            while (cdf[j] < u) { ++j; }
            x2.col(i) = x.col(j);
        }
        x = x2;
        w.setConstant(w0);
    }

    void invalidate() {
        valid = false;
        eta = 0;
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

        for (unsigned j = 0; j < pdf_weights.cols(); ++j) {
            pdfs[j].prepare_sort();
            q.emplace(pdf_weights[j] * pdfs[j].wsorted(0), j, 0);
        }

        for (unsigned i = 0; i < N; ++i) {
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
        pdf.resample();
    }

    State mean() const {
        return (x.array().rowwise() * w).rowwise().sum();
    }

    Covariance cov(const State& m) const {
        // See https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance
        auto d = (x.colwise() - m.array());
        return (d.rowwise() * w).matrix() * d.matrix().transpose()
            // Unbiased:
            // (1 - (w * w).sum())
        ;
    }

    Covariance cov() const {
        return cov(mean());
    }

    Eigen::Vector2d pos() {
        return mean().template head<2>();
    }

    Eigen::Matrix2d poscov() const {
        return cov().template topLeftCorner<2, 2>();
    }

    AABBox llaabbox() {
        return _llaabbox;
    }

    void transform_to_local(const cf::LL& origin_) {
        origin = origin_;
        for (unsigned i = 0; i < N; ++i) {
            cf::ll2ne_i(x.col(i).template head<2>(), origin);
        }
    }

    cf::LL transform_to_local() {
        transform_to_local(pos());
        return origin;
    }

    void transform_to_global() {
        update_local_bbox();
        for (unsigned i = 0; i < N; ++i) {
            cf::ne2ll_i(x.col(i).template head<2>(), origin);
        }
        bbox = bbox.llbbox(origin);
        aabbox = bbox.aabbox();
    }
};

template<unsigned STATES, unsigned PARTICLES = 200>
auto& operator<<(std::ostream& os, const PF<STATES, PARTICLES>& t) {
    t.repr(os);
    return os;
}
}  // namespace lmb
