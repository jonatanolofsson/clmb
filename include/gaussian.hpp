// Copyright 2018 Jonatan Olofsson
#pragma once
#include <Eigen/Core>
#include <vector>
#include "bbox.hpp"
#include "cf.hpp"
#include "statistics.hpp"
#include "omp.hpp"

namespace lmb {

template<int S>
struct alignas(16) Gaussian_ {
    static const int STATES = S;
    using Self = Gaussian_<S>;
    using State = Eigen::Matrix<double, STATES, 1>;
    using Covariance = Eigen::Matrix<double, STATES, STATES>;
    State x;
    Covariance P;
    double w;

    Gaussian_() {}

    Gaussian_(const State& x_, const Covariance& P_, double w_ = 1.0)
    : x(x_),
      P(P_),
      w(w_)
    {}

    template<typename Sensor>
    double correct(const typename Sensor::Report& z, const Sensor& s, const cf::LL& origin) {
        auto dz = (z.mean() - s.measurement(x)).eval();
        auto Dinv = (s.H * P * s.H.transpose() + z.cov()).inverse().eval();
        auto K = P * s.H.transpose() * Dinv;
        //std::cout << "pD: " << s.pD(*this, origin) << std::endl;
        //std::cout << "likelihood: " << z.likelihood(dz, Dinv) << std::endl;
        w *= s.pD(*this, origin) * z.likelihood(dz, Dinv) / z.kappa;
        x += K * dz;
        P -= K * s.H * P;

        return w;
    }

    bool operator<(const Gaussian_& b) const {
        return w < b.w;
    }

    State mean() const {
        return x;
    }

    Covariance cov() const {
        return P;
    }

    Eigen::Vector2d pos() const {
        return x.template head<2>();
    }

    Eigen::Matrix2d poscov() const {
        return P.template topLeftCorner<2, 2>();
    }

    void set_state(const State& state) {
        x = state;
    }

    template<typename FT, typename QT>
    void linear_update(const FT& F, const QT& Q) {
        //std::cout << "F: " << std::endl << F << std::endl;
        //std::cout << "Q: " << std::endl << Q << std::endl;
        //std::cout << "Before update: " << x.format(eigenformat) << std::endl;
        //std::cout << P << std::endl;
        x = F * x;
        P = F * P * F.transpose() + Q;
        //std::cout << "After update: " << x.format(eigenformat) << std::endl;
        //std::cout << P << std::endl;
    }

    BBox bbox(const double nstd = 2.0) const {
        return BBox(pos(), poscov(), nstd);
    }

    AABBox aabbox(const double nstd = 2.0) const {
        return AABBox(pos(), poscov(), nstd);
    }

    BBox nebbox(const cf::LL& origin, const double nstd = 2.0) const {
        return BBox(cf::ll2ne(pos(), origin), poscov(), nstd);
    }

    template<typename RES>
    void sample(RES& res) const {
        nrand(res, x, P);
    }

    void sampled_pos_pdf(const Eigen::Array<double, 2, Eigen::Dynamic>& points,
                         Eigen::Array<double, 1, Eigen::Dynamic>& res,
                         const double scale = 1) const {
        const double logSqrt2Pi = 0.5*std::log(2*M_PI);
        typedef Eigen::LLT<Eigen::Matrix2d> Chol;
        Chol chol(poscov());
        if (chol.info() != Eigen::Success) {
            throw "decomposition failed!";
        }
        const Chol::Traits::MatrixL& L = chol.matrixL();
        auto diff = (points.matrix().colwise() - pos()).matrix().eval();
        auto quadform = L.solve(diff).colwise().squaredNorm().array();
        auto pdf = ((-0.5*quadform - points.rows()*logSqrt2Pi).exp()
            / L.determinant()).eval();
        double psum = pdf.sum();
        if (psum > 1e-7) {
            res.array() += pdf * scale / psum;
        }
    }

    double overlap(const BBox& bbox) const {
        static const int N = 10000;  // FIXME
        Eigen::Matrix<double, S, Eigen::Dynamic> samples(S, N);
        sample(samples);
        int nwithin = 0;
        PARFOR
        for (int n = 0; n < N; ++n) {
            if (bbox.within(samples.col(n))) {
                ++nwithin;
            }
        }
        return static_cast<double>(nwithin) / N;
    }

    void repr(std::ostream& os) const {
        os << "{\"type\":\"G\","
            << "\"w\":" << w
            << ",\"x\":" << x.format(eigenformat)
            << ",\"P\":" << P.format(eigenformat)
            << "}";
    }

    void transform_to_local(const cf::LL& origin) {
        cf::ll2ne_i(x.template head<2>(), origin);
    }

    void transform_to_global(const cf::LL& origin) {
        cf::ne2ll_i(x.template head<2>(), origin);
    }
};

template<int S>
auto& operator<<(std::ostream& os, const Gaussian_<S> c) {
    c.repr(os);
    return os;
}
}  // namespace lmb
