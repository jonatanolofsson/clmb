// Copyright 2018 Jonatan Olofsson
#pragma once
#include <algorithm>
#include <iostream>
#include <vector>
#include "bbox.hpp"
#include "cf.hpp"
#include "gaussian.hpp"
#include "params.hpp"

namespace lmb {

enum Action {
    ACTION_PREDICT = 'u',
    ACTION_CORRECT = 'c',
    ACTION_INIT = 'i'
};

template<int S>
struct alignas(16) TargetSummary_ : public Gaussian_<S> {
    using Self = TargetSummary_<S>;
    using Parent = Gaussian_<S>;
    using State = typename Parent::State;
    using Covariance = typename Parent::Covariance;
    double& r;
    unsigned id;
    unsigned cid;

    TargetSummary_(const State& x_, const Covariance P_,
                   double w_, unsigned id_, unsigned cid_ = 0)
    : Parent(x_, P_, w_), r(this->w), id(id_), cid(cid_) {}

    void repr(std::ostream& os) const {
        os << "{\"type\":\"T\""
           << ",\"id\":" << id
           << ",\"w\":" << this->w
           << ",\"x\":" << this->x.format(eigenformat)
           << ",\"P\":" << this->P.format(eigenformat)
           << "}";
    }
};
template<int S> using TargetSummaries_ = std::vector<TargetSummary_<S>>;

template<typename PDF_>
struct Target_ {
    using PDF = PDF_;
    using Self = Target_<PDF>;
    using Targets = std::vector<Self*>;
    using Gaussian = Gaussian_<PDF::STATES>;
    using TargetSummary = TargetSummary_<PDF::STATES>;
    unsigned id;
    double r;
    PDF pdf;
    std::vector<PDF> pdfs;
    unsigned cluster_id;
    const Params* params;
    bool is_new = true;
    double t0 = 0;

    Action last_action;

    Target_()
    : r(0),
      pdfs(0),
      cluster_id(0),
      params(nullptr),
      last_action(ACTION_INIT)
    {}

    Target_(double r_, PDF&& pdf_, const Params* params_, double t0_)
    : r(r_),
      pdf(pdf_),
      pdfs(0),
      cluster_id(0),
      params(params_),
      t0(t0_),
      last_action(ACTION_INIT)
    {}

    bool viable() {
        if (r < params->r_lim) {
            //std::cout << "Removing b/c r_lim" << std::endl;
            return false;
        }
        if (pdf.poscov().norm() > params->cov_lim) {
            //std::cout << "Removing b/c poscov" << std::endl;
            return false;
        }
        return true;
    }

    void transform_to_local(const cf::LL& origin) {
        pdf.transform_to_local(origin);
    }

    cf::LL transform_to_local() {
        return pdf.transform_to_local();
    }

    void transform_to_global() {
        pdf.transform_to_global();
    }

    cf::LL pos() {
        return pdf.pos();
    }

    explicit Target_(const Self&) = delete;
    Self operator=(const Self&) = delete;

    template<typename Sensor, typename RES, typename MRES>
    void match(const std::vector<typename Sensor::Report*>& reports,
               const Sensor& sensor,
               RES&& res,
               MRES& mres,
               const double time) {
        unsigned M = reports.size();
#ifdef DEBUG_OUTPUT
        if (cluster_id == 0) {
        std::cout << "Match against: " << pdf << std::endl;
        }
#endif
        pdfs.clear();
        pdfs.resize(M + 1, pdf);

        PARFOR
        for (unsigned j = 0; j < M; ++j) {
            if (is_new) { sensor.pdf_init2(pdfs[j], *reports[j], time, t0); }
            res(0, j) = -std::log(r * pdfs[j].correct(*reports[j], sensor));
        }
        is_new = false;
#ifdef DEBUG_OUTPUT
        if (cluster_id == 0) {
        std::cout << "Candidate corrections: " << std::endl;
        for (unsigned j = 0; j < M + 1; ++j) {
            std::cout << "\t" << pdfs[j] << std::endl;
        }
        }
#endif
        mres = -std::log(r * pdfs[M].missed(sensor));
    }

    template<typename Points, typename RES>
    void distance(const Points& points, RES&& res) const {
        unsigned M = points.size();
        auto m = pdf.mean();

        for (unsigned j = 0; j < M; ++j) {
            res(0, j) = (points[j] - m).norm();
        }
    }

    AABBox llaabbox() {
        return pdf.llaabbox();
    }

    AABBox neaabbox(const cf::LL& origin) {
        return llaabbox().neaabbox(origin);
    }

    template<typename W>
    void correct(const W& w, double rlim = 1e-6) {
        r = w.sum();
        if (r >= rlim) {
            PDF::join(pdf, w, pdfs);
            last_action = ACTION_CORRECT;
        } else {
            r = 0;
        }
    }

    template<typename Model>
    void predict(Model& model, const double time, const double last_time) {
#ifdef DEBUG_OUTPUT
        std::cout << "dT: " << time - last_time << std::endl;
#endif
        model(this, time, last_time);
        pdf.normalize();
        r *= pdf.eta;
        last_action = ACTION_PREDICT;
    }

    double false_target() {
        if (r < 1) {
            return -std::log(1 - r);
        } else {
            return inf;
        }
    }

    void pos_phd(const Eigen::Array<double, 2, Eigen::Dynamic>& points,
                 Eigen::Array<double, 1, Eigen::Dynamic>& res) const {
        pdf.sampled_pos_pdf(points, res, r);
    }

    Gaussian summary() {
        return TargetSummary(pdf.mean(), pdf.cov(), r, id, cluster_id);
    }

    void repr(std::ostream& os) const {
        os << "{\"id\":" << id
           << ",\"cid\":" << cluster_id
           << ",\"la\":\"" << char(last_action) << "\""
           << ",\"r\":" << r
           << ",\"pdf\":" << pdf << "}";
    }
};

template<typename PDF> using Targets_ = typename Target_<PDF>::Targets;

template<typename PDF>
auto& operator<<(std::ostream& os, const Target_<PDF>& t) {
    t.repr(os);
    return os;
}

}  // namespace lmb
