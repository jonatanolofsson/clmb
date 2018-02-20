// Copyright 2018 Jonatan Olofsson
#pragma once
#include <Eigen/Core>
#include <vector>
#include "bbox.hpp"
#include "cf.hpp"
#include "gaussian.hpp"

namespace lmb {

template<int S>
struct alignas(16) GaussianReport_ : public Gaussian_<S> {
    using Self = GaussianReport_<S>;
    using Gaussian = Gaussian_<S>;
    using State = typename Gaussian::State;
    using Covariance = typename Gaussian::Covariance;

    volatile double rB;
    double kappa;

    GaussianReport_(const State x_, const Covariance P_, double kappa_)
    : Gaussian(x_, P_) {
        kappa = kappa_;
    }

    template<typename DZ, typename DINV>
    double likelihood(const DZ& dz, const DINV& Dinv) const {
        return std::exp(-(dz.transpose() * Dinv * dz)(0) * 0.5)
            * std::sqrt(Dinv.determinant() / std::pow(2 * M_PI, dz.size()));
    }

    void repr(std::ostream& os) const {
        os << "{\"type\":\"GR\""
           << ",\"x\":" << this->x.format(eigenformat)
           << ",\"P\":" << this->P.format(eigenformat) << "}";
    }

    AABBox llaabbox() {
        return AABBox(this->P).llaabbox(this->pos());
    }
};

template<int S>
auto& operator<<(std::ostream& os, const GaussianReport_<S>& r) {
    r.repr(os);
    return os;
}

template<int S>
auto& operator<<(std::ostream& os, const std::vector<GaussianReport_<S>>& reports) {  // NOLINT
    os << "[";
    bool first = true;
    for (auto& r : reports) {
        if (!first) { os << ","; } else { first = false; }
        os << r;
    }
    os << "]";
    return os;
}

}  // namespace lmb
