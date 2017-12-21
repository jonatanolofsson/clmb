#pragma once
#include "bbox.hpp"
#include "gauss.hpp"
#include <iostream>

namespace lmb {
    enum Action {
        ACTION_PREDICT = 'u',
        ACTION_CORRECT = 'c',
        ACTION_INIT = 'i'
    };

    template<typename PDF_>
    struct Target {
        typedef PDF_ PDF;
        typedef Target<PDF> Self;
        typedef std::vector<Self*> Targets;
        typedef GaussianComponent<PDF::STATES> Gaussian;
        unsigned id;
        double r;
        PDF pdf;
        std::vector<PDF> pdfs;
        unsigned cluster;
        double t;
        Action last_action;

        Target()
        : r(0),
          pdfs(0),
          cluster(0),
          t(0.0),
          last_action(ACTION_INIT)
        {}

        Target(double r_, PDF&& pdf_)
        : r(r_),
          pdf(pdf_),
          pdfs(0),
          cluster(0),
          t(0.0),
          last_action(ACTION_INIT)
        {}

        Target(const Self&) = delete;
        Self operator=(const Self&) = delete;

        template<typename Report, typename Sensor, typename RES, typename MRES>
        void match(const std::vector<Report*>& reports, const Sensor& sensor, RES&& res, MRES& mres) {
            unsigned M = reports.size();
            pdfs.clear();
            pdfs.resize(M + 1, pdf);

            for (unsigned j = 0; j < M; ++j) {
                res(0, j) = -std::log(r * pdfs[j].correct(*reports[j], sensor));
            }
            mres = -std::log(r * pdfs[M].missed(sensor));
        }

        template<typename W>
        void correct(const W& w, double rlim=1e-6) {
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

        Gaussian gauss_state() {
            return Gaussian(r, pdf.mean(), pdf.cov());
        }

        void repr(std::ostream& os) const {
            os << "{\"id\":" << id
               << ",\"cid\":" << cluster
               << ",\"la\":\"" << char(last_action) << "\""
               << ",\"t\":" << t
               << ",\"r\":" << r
               << ",\"pdf\":" << pdf << "}";
        }
    };

    template<typename PDF>
    auto& operator<<(std::ostream& os, const Target<PDF>& t) {
        t.repr(os);
        return os;
    }

    template<typename PDF> using Targets = std::vector<Target<PDF>*>;

    template<typename Report, typename Target>
    struct Cluster {
        std::vector<Report*> reports;
        std::vector<Target*> targets;

        Cluster(const std::vector<Report*>& reports_, const std::vector<Target*>& targets_)
        : reports(reports_),
          targets(targets_)
        {}
    };
    template<typename Report, typename Target> using Clusters = std::vector<Cluster<Report, Target>>;
}
