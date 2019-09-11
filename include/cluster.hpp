// Copyright 2018 Jonatan Olofsson
#pragma once
#include <Eigen/Eigenvalues>
#include <cmath>
#include <vector>
#include "cf.hpp"
#include "murty.hpp"
#include "params.hpp"
#include "report.hpp"
#include "target.hpp"

namespace lmb {
using lap::Murty;
using lap::Assignment;

template<typename Report, typename Target>
struct Cluster_ {
    using Self = Cluster_;
    using Reports = std::vector<Report*>;
    using Targets = std::vector<Target*>;
    cf::LL origin;

    Reports reports;
    Targets targets;
    unsigned id;
    unsigned n;
    const Params* params;

    Cluster_(const unsigned id_, const Reports& reports_, const Targets& targets_, const Params* params_)  // NOLINT
    : reports(reports_),
      targets(targets_),
      id(id_),
      params(params_)
    {}

    template<typename TargetTree, typename Sensor>
    void correct(TargetTree& targettree, const Sensor& sensor, const double time) {
        unsigned M = targets.size();
        unsigned N = reports.size();
        n = 0;

        // Remove from tree while updating
        CRITICAL(ttree)
        {
            for (unsigned i = 0; i < M; ++i) {
                targettree.remove(targets[i]);
            }
        }

        // Transform to NED coordinates
#ifdef DEBUG_OUTPUT
        if (id == 0) {
        std::cout << "Pre-local: " << *this << std::endl;
        }
#endif
        transform_to_local();
#ifdef DEBUG_OUTPUT
        if (id == 0) {
        std::cout << "Post-local: " << *this << std::endl;
        }
#endif

        if (M > 0) {
            // There are targets to match with
            // Create cost matrix
            Eigen::MatrixXd C(M, N + 2 * M);
            C.setConstant(inf);
            for (unsigned i = 0; i < M; ++i) {
                targets[i]->match(reports,
                                  sensor,
                                  C.block(i, 0, 1, N),
                                  C(i, N + i),
                                  time);
                C(i, N + M + i) = targets[i]->false_target();
            }

#ifdef DEBUG_OUTPUT
            if (id == 0) {
            std::cout << "C: \n" << C << std::endl;
            }
#endif

            Murty murty(C);
            Assignment res;
            double cost;
            double w;
            double w_sum = 0;
            Eigen::MatrixXd R(M, N + 1);
            R.setZero();
            // Draw most relevant hypotheses using Murty's algorithm
            while (murty.draw(res, cost)) {
                w = std::exp(-cost);
                w_sum += w;
                #pragma omp parallel for
                for (unsigned i = 0; i < M; ++i) {
                    if ((unsigned)res[i] < N) {
                        R(i, res[i]) += w;
                    } else if ((unsigned)res[i] == N + i) {
                        R(i, N) += w;
                    }
                }
                ++n;
                if (w / w_sum < params->w_lim || n >= params->nhyp_max) {
                    break;
                }
            }
            if (n > 0) {
                R /= w_sum;

                // Update each target according to the hypotheses' associations
                for (unsigned i = 0; i < M; ++i) {
                    targets[i]->correct(R.row(i));
                }

                // Forward birth probabilities to birth algorithm
                auto rB = (1.0 - R.colwise().sum().array()).eval();
                for (unsigned j = 0; j < N; ++j) {
                    reports[j]->rB = rB[j];
                }
            }
        }

        // No targets to match with, or no valid hypotheses?
        if (n == 0) {
            for (auto& r : reports) {
                r->rB = 1;
            }
        }

        // Move back to LL CF again
#ifdef DEBUG_OUTPUT
        if (id == 0) {
        std::cout << "Pre-global: " << *this << std::endl;
        }
#endif
        transform_to_global();
#ifdef DEBUG_OUTPUT
        if (id == 0) {
        std::cout << "Post-global: " << *this << std::endl;
        }
#endif

        // Replace the still valid targets into tree
        CRITICAL(ttree)
        {
            for (unsigned i = 0; i < M; ++i) {
                if (targets[i]->viable()) {
                    targettree.replace(targets[i]);
                } else {
                    targettree.erase(targets[i]);
                }
            }
        }
    }

    cf::LL center_of_mass() {
        cf::LL s; s.setZero();
        for (auto r : reports) { s += r->pos(); }
        for (auto t : targets) { s += t->pos(); }
        auto n = reports.size() + targets.size();
        return s / n;
    }

    void transform_to_local() {
#ifdef DEBUG_OUTPUT
        std::cout << "Transforming cluster " << id << " to local" << std::endl;
#endif
        origin = center_of_mass();
#ifdef DEBUG_OUTPUT
        std::cout << "  Origin: " << origin.format(eigenformat) << std::endl;
        std::cout << "Reports: " << std::endl;
#endif
        for (auto& r : reports) {
#ifdef DEBUG_OUTPUT
            std::cout << "\t" << *r;
#endif
            r->transform_to_local(origin);
#ifdef DEBUG_OUTPUT
            std::cout << " -> " << *r << std::endl;
#endif
        }
#ifdef DEBUG_OUTPUT
        std::cout << "Targets: " << std::endl;
#endif
        for (auto& t : targets) {
#ifdef DEBUG_OUTPUT
            std::cout << "\t" << *t;
#endif
            t->transform_to_local(origin);
#ifdef DEBUG_OUTPUT
            std::cout << " -> " << *t << std::endl;
#endif
        }
    }

    void transform_to_global() {
#ifdef DEBUG_OUTPUT
        std::cout << "Transforming cluster " << id << " to global" << std::endl;
        std::cout << "  Origin: " << origin.format(eigenformat) << std::endl;
        std::cout << "Reports: " << std::endl;
#endif
        for (auto& r : reports) {
#ifdef DEBUG_OUTPUT
            std::cout << "\t" << *r;
#endif
            r->transform_to_global(origin);
#ifdef DEBUG_OUTPUT
            std::cout << " -> " << *r << std::endl;
#endif
        }
#ifdef DEBUG_OUTPUT
        std::cout << "Targets: " << std::endl;
#endif
        for (auto& t : targets) {
#ifdef DEBUG_OUTPUT
            std::cout << "\t" << *t;
#endif
            t->transform_to_global();
#ifdef DEBUG_OUTPUT
            std::cout << " -> " << *t << std::endl;
#endif
        }
    }

    void repr(std::ostream& os) const {
        os << "{\"type\":\"CL\"";
        os << ",\"R\":[";
        bool first = true;
        for (auto& r : reports) {
            if (!first) { os << ","; } else { first = false; }
            os << *r;
        }
        os << "],\"T\":[";
        first = true;
        for (auto& t : targets) {
            if (!first) { os << ","; } else { first = false; }
            os << *t;
        }
        os << "]}";
    }
};

template<typename Report, typename Target>
auto& operator<<(std::ostream& os, const Cluster_<Report, Target>& f) {
    f.repr(os);
    return os;
}

}  // namespace lmb
