#pragma once
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <algorithm>
#include "bbox.hpp"
#include "constants.hpp"

namespace lmb {
    struct alignas(16) Report {
        typedef Report Self;
        typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Measurement;
        typedef Eigen::MatrixXd Covariance;
        const Measurement z;
        const Covariance& R;
        const double kappa;
        volatile double rB;
        AABBox aabbox;

        Report(const Measurement& z_, const Covariance& R_, const double kappa_)
        : z(z_),
          R(R_),
          kappa(kappa_),
          aabbox(z, R)
        {}

        void reset() {
            aabbox = AABBox(z, R);
        }

        template<typename DZ, typename DINV>
        double likelihood(const DZ& dz, const DINV& Dinv) const {
            return std::exp(-(double)(dz.transpose() * Dinv * dz) * 0.5) * std::sqrt(Dinv.determinant() / std::pow(2 * M_PI, dz.size()));
        }

        void dump(std::ostream& os) const {
            os << "{\"z\":" << z.format(eigenformat) << ",\"R\":" << R.format(eigenformat) << "}";
        }
    };

    auto& operator<<(std::ostream& os, const Report& t) {
        t.dump(os);
        return os;
    }

    auto& operator<<(std::ostream& os, const std::vector<Report>& reports) {
        os << "[";
        bool first = true;
        for (auto& r : reports) {
            if (!first) { os << ","; } else { first = false; }
            os << r;
        }
        os << "]";
        return os;
    }

    template<typename Target>
    struct PositionSensor {
        typedef Eigen::Matrix<double, 2, Target::PDF::STATES> ObservationMatrix;
        inline static const ObservationMatrix H = ObservationMatrix::Identity();
        AABBox aabbox;
        BBox fov;
        Eigen::Matrix2d pv;

        PositionSensor()
        : aabbox(-inf, -inf, inf, inf),
          pv(Eigen::Matrix2d::Identity() * 10)
        {}

        void set_fov(const BBox& f) {
            fov = f;
            aabbox = fov.aabbox();
        }

        template<typename States>
        auto measurement(const States& m) const {
            return H * m;
        }

        template<typename T>
        double pD(const T&) const {
            // FIXME
            return 1.0;
        }

        template<typename Targets>
        void fov_filter(Targets& targets) const {
            Targets result;
            result.reserve(targets.size());
            std::copy_if(targets.begin(), targets.end(), result.begin(),
                         [this](typename Targets::value_type t) { return fov.intersects(t->pdf.aabbox); });
            targets.swap(result);
        }

        typename Target::PDF pdf(const Report& r) const {
            // FIXME: Better initial covariance (Rasmussen, Williams?)
            typename Target::PDF::Mean m;
            typename Target::PDF::Covariance P;
            m << r.z[0], r.z[1], 0.0, 0.0;
            P.setZero();
            P.template block<2, 2>(0, 0) = r.R;
            P.template block<2, 2>(2, 2) = pv;
            return typename Target::PDF(m, P);
        }
    };
}
