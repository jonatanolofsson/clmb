#pragma once
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <algorithm>
#include "bbox.hpp"
#include "constants.hpp"
#include "target.hpp"
#include "targettree.hpp"

namespace lmb {
    struct alignas(16) GaussianReport {
        using Self = GaussianReport;
        using Measurement = Eigen::Matrix<double, Eigen::Dynamic, 1>;
        using Covariance = Eigen::MatrixXd;
        Measurement z;
        Covariance R;
        double kappa;
        volatile double rB;
        AABBox aabbox;

        GaussianReport(const Measurement z_, const Covariance R_, double kappa_)
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

        void repr(std::ostream& os) const {
            os << "{\"z\":" << z.format(eigenformat) << ",\"R\":" << R.format(eigenformat) << "}";
        }
    };

    auto& operator<<(std::ostream& os, const GaussianReport& r) {
        r.repr(os);
        return os;
    }

    auto& operator<<(std::ostream& os, const std::vector<GaussianReport>& reports) {
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
        using Self = PositionSensor<Target>;
        using Report = GaussianReport;
        using ObservationMatrix = Eigen::Matrix<double, 2, Target::PDF::STATES>;
        using Targets = Targets_<typename Target::PDF>;
        inline static const ObservationMatrix H = ObservationMatrix::Identity();
        BBox fov;
        Eigen::Matrix2d pv;
        double lambdaB = 1.0;
        double pD_ = 1.0;

        PositionSensor()
        : pv(Eigen::Matrix2d::Identity() * 10)
        {}

        template<typename States>
        auto measurement(const States& m) const {
            return H * m;
        }

        template<typename PDF>
        double pD(const PDF& pdf) const {
            return pD_ * pdf.overlap(fov);
        }

        Targets get_targets(const TargetTree<typename Target::PDF>& targettree) const {
            Targets result;
            auto targets = targettree.query(fov.aabbox());
            result.reserve(targets.size());
            std::copy_if(targets.begin(), targets.end(), result.begin(),
                         [this](typename Targets::value_type t) { return t->pdf.intersects(fov); });
            result.shrink_to_fit();
            return result;
        }

        template<typename Report>
        typename Target::PDF pdf(Params* params, const Report& r) const {
            // FIXME: Better initial covariance (Rasmussen, Williams?)
            typename Target::PDF::State m;
            typename Target::PDF::Covariance P;
            m << r.z[0], r.z[1], 0.0, 0.0;
            P.setZero();
            P.template block<2, 2>(0, 0) = r.R;
            P.template block<2, 2>(2, 2) = pv;
            return typename Target::PDF(params, m, P);
        }
    };
}
