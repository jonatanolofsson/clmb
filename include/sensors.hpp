// Copyright 2018 Jonatan Olofsson
#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <vector>
#include "bbox.hpp"
#include "constants.hpp"
#include "params.hpp"
#include "report.hpp"
#include "target.hpp"
#include "targettree.hpp"

namespace lmb {
template<typename Target>
struct PositionSensor {
    using Self = PositionSensor<Target>;
    using Report = GaussianReport_<2>;
    using ObservationMatrix = Eigen::Matrix<double, 2, Target::PDF::STATES>;
    using Targets = Targets_<typename Target::PDF>;
    using Scan = std::vector<Report>;
    using TargetTree = TargetTree_<typename Target::PDF>;
    inline static const ObservationMatrix H = ObservationMatrix::Identity();
    BBox fov;
    Eigen::Matrix2d pv;
    double lambdaB = 1.0;
    double pD_ = 1.0;
    double kappa_ = 1.0;

    PositionSensor()
    : pv(Eigen::Matrix2d::Identity())
    {
        fov.corners << 90, 90, -90, -90,
                       180, 0,  0,  180;
        //fov.corners << 90, 90, -90, -90,
                       //360, 0,  0,  360;
    }

    template<typename States>
    auto measurement(const States& m) const {
        return H * m;
    }

    template<typename PDF>
    double pD(const PDF& pdf, const cf::LL& origin) const {
        return pD_ * pdf.overlap(fov.nebbox(origin));
    }

    Targets get_targets(const TargetTree& targettree) const {
        Targets result;
        typename TargetTree::Targets targets;
        CRITICAL(ttree)
        {
            targets = targettree.query(fov.aabbox());
        }
        result.reserve(targets.size());
        //std::cout << "Queried nof targets: " << targets.size() << std::endl;
        std::copy_if(targets.begin(), targets.end(), std::back_inserter(result),
                     [this](typename Targets::value_type t) {
                        return t->pdf.intersects(fov);
                     });
        //std::cout << "Results: " << result.size() << std::endl;
        result.shrink_to_fit();
        return result;
    }

    template<typename Report>
    typename Target::PDF pdf_init1(Params* params, const Report& r, const double = 0) const {
        // FIXME: Better initial velocity (Rasmussen, Williams?)
        typename Target::PDF::State x;
        typename Target::PDF::Covariance P;
        x << r.mean().template head<2>(), 0.0, 0.0;
        P.setZero();
        P.template block<2, 2>(0, 0) = r.cov();
        P.template block<2, 2>(2, 2) = pv;
        return typename Target::PDF(params, x, P);
    }

    template<typename Report>
    void pdf_init2(typename Target::PDF& pdf, const Report& z, const double time, const double t0) const {
        return;
        const double dT = (time - t0) / pdf.params->tscale;
        if (dT < 1e-5) { std::cout << "init2 failed: " << dT << std::endl; return; }
        auto x = pdf.mean();
        std::cout << "Secondary init (" << dT << ") using " << z.mean().format(eigenformat) << ": " << x.format(eigenformat);
        x.template tail<2>() = (z.mean() - x.template head<2>()) / dT;
        Eigen::Matrix4d F;
        F << 1, 0, dT, 0,
             0, 1, 0, dT,
             0, 0, 1, 0,
             0, 0, 0, 1;
        x = F * x;
        std::cout << " -> " << x.format(eigenformat) << std::endl;
        pdf.set_state(x);
    }

    template<typename Report>
    double kappa(const Report&) const {
        return kappa_;
    }
};
}  // namespace lmb
