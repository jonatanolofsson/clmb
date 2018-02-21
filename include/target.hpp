// Copyright 2018 Jonatan Olofsson
#pragma once
#include <algorithm>
#include <iostream>
#include <vector>
#include "bbox.hpp"
#include "cf.hpp"
#include "gaussian.hpp"

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
                   double w_, unsigned id_, unsigned cid_=0)
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

    double t;
    Action last_action;

    Target_()
    : r(0),
      pdfs(0),
      cluster_id(0),
      t(0.0),
      last_action(ACTION_INIT)
    {}

    Target_(double r_, PDF&& pdf_)
    : r(r_),
      pdf(pdf_),
      pdfs(0),
      cluster_id(0),
      t(0.0),
      last_action(ACTION_INIT)
    {}

    void transform_to_local(const cf::LL& origin) {
        pdf.transform_to_local(origin);
    }

    cf::LL transform_to_local() {
        auto origin = pos();
        transform_to_local(origin);
        return origin;
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
               MRES& mres) {
        unsigned M = reports.size();
        pdfs.clear();
        pdfs.resize(M + 1, pdf);

        for (unsigned j = 0; j < M; ++j) {
            res(0, j) = -std::log(r * pdfs[j].correct(*reports[j], sensor));
        }
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

    template<typename W>
    void correct(const W& w, double rlim = 1e-6) {
        r = w.sum();
        if (r >= rlim) {
            PDF::join(pdf, w, pdfs);
            last_action = ACTION_CORRECT;
        } else {
            r = -inf;
        }
    }

    template<typename Model>
    void predict(Model& model, double time) {
        model(this, time);
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

    Gaussian summary() {
        return TargetSummary(pdf.mean(), pdf.cov(), r, id, cluster_id);
    }

    void repr(std::ostream& os) const {
        os << "{\"id\":" << id
           << ",\"cid\":" << cluster_id
           << ",\"la\":\"" << char(last_action) << "\""
           << ",\"t\":" << t
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
