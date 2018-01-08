// Copyright 2018 Jonatan Olofsson
#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include "bbox.hpp"
#include "connectedcomponents.hpp"
#include "lap.hpp"
#include "murty.hpp"
#include "omp.hpp"
#include "params.hpp"
#include "sensors.hpp"
#include "target.hpp"
#include "targettree.hpp"

namespace lmb {

template<typename PDF_>
class SILMB {
 public:
    using PDF = PDF_;
    using Self = SILMB<PDF>;
    using Target = Target_<PDF>;
    using Targets = Targets_<PDF>;
    using Gaussian = Gaussian_<PDF::STATES>;
    using TargetStates = std::vector<typename PDF::State>;
    using TargetSummaries = TargetSummaries_<PDF::STATES>;


    Params params;

    TargetTree<PDF> targettree;

    SILMB() {}

    explicit SILMB(const Params& params_) : params(params_) {}


    template<typename Model>
    void predict(Model& model, double time) {
        targettree.lock();
        for (auto& t : targettree.targets) {
            targettree.remove(t);
        }
        targettree.unlock();

        PARFOR
        for (auto t = std::begin(targettree.targets);
                t < std::end(targettree.targets); ++t) {
            (*t)->template predict<Model>(model, time);
        }

        targettree.lock();
        for (auto& t : targettree.targets) {
            targettree.replace(t);
        }
        targettree.unlock();
    }

    template<typename Model>
    void predict(Model& model, const AABBox& aabbox, double time) {
        auto all_targets = targettree.query(aabbox);

        targettree.lock();
        for (auto& t : all_targets) {
            targettree.remove(t);
        }
        targettree.unlock();

        PARFOR
        for (auto t = std::begin(all_targets); t < std::end(all_targets); ++t) {
            (*t)->template predict<Model>(model, time);
        }

        targettree.lock();
        for (auto& t : all_targets) {
            targettree.replace(t);
        }
        targettree.unlock();
    }

    template<typename Sensor>
    void correct(std::vector<typename Sensor::Report>& reports, const Sensor& sensor, double time) {
        using Report = typename Sensor::Report;
        Clusters<Report> clusters;
        auto all_targets = sensor.get_targets(targettree);

        cluster(reports, all_targets, clusters);

        PARFOR
        for (auto c = std::begin(clusters); c < std::end(clusters); ++c) {
            cluster_correct<Report, Sensor>(*c, sensor);
        }

        birth(reports, sensor, time);
    }

    double enof_targets() {
        double res = 0;
        targettree.lock();
        for (auto& t : targettree.targets) {
            res += t->r;
        }
        targettree.unlock();
        return res;
    }

    unsigned nof_targets(const double r_lim) {
        unsigned res = 0;
        targettree.lock();
        for (auto& t : targettree.targets) {
            if (t->r >= r_lim) { ++res; }
        }
        targettree.unlock();
        return res;
    }

    TargetSummaries get_targets() {
        TargetSummaries res;
        targettree.lock();
        for (auto& t : targettree.targets) {
            res.emplace_back(t->pdf.mean(), t->pdf.cov(), t->r, t->id);
        }
        targettree.unlock();
        return res;
    }

    double ospa(const TargetStates& truth, const double c, const double p) {
        targettree.lock();
        int M = targettree.targets.size();
        int N = truth.size();
        int n = std::max(M, N);

        if (n == 0) {
            return 0;
        }

        Eigen::MatrixXd C(n, n);
        C.setConstant(c);
        for (int i = 0; i < M; ++i) {
            targettree.targets[i]->distance(truth, C.block(i, 0, 1, N));
        }
        C = C.cwiseMin(c);
        targettree.unlock();

        // Note that since exp() is strictly increasing, C and C.^p yields the
        // same assignments. By only raising the result, we compute fewer
        // exponentials
        auto res = lap::lap(C);

        double cost = 0;
        for (unsigned i = 0; i < res.rows(); ++i) {
            cost += std::pow(C(i, res[i]), p);
        }
        return std::pow(cost / n, 1.0 / p);
    }

    double gospa(const TargetStates& truth, const double c, const double p) {
        targettree.lock();
        int M = targettree.targets.size();
        int N = truth.size();
        int n = std::max(M, N);

        if (n == 0) {
            return 0;
        }

        double cp = std::pow(c, p);

        Eigen::MatrixXd C(n, n);
        C.setConstant(cp / 2);
        for (int i = 0; i < M; ++i) {
            targettree.targets[i]->distance(truth, C.block(i, 0, 1, N));
        }
        C.block(0, 0, M, N).array() =
            C.block(0, 0, M, N).cwiseMin(c).array().pow(p);
        targettree.unlock();

        auto res = lap::lap(C);
        return std::pow(lap::cost(C, res), 1.0 / p);
    }

    void repr(std::ostream& os) const {
        os << "{\"targets\":[";
        bool first = true;
        for (auto& t : targettree.targets) {
            if (!first) { os << ","; } else { first = false; }
            os << *t;
        }
        os << "]}";
    }

 private:
    template<typename Report>
    struct Cluster {
        using Self = Cluster<Report>;
        using Clusters = std::vector<Self>;
        using Reports = std::vector<Report*>;
        using Targets = std::vector<Target*>;

        Reports reports;
        Targets targets;

        Cluster(const Reports& reports_, const Targets& targets_)
        : reports(reports_),
          targets(targets_)
        {}
    };

    template<typename Report> using Clusters =
        typename Cluster<Report>::Clusters;

    template<typename Report>
    void cluster(std::vector<Report>& reports,
                 Targets& all_targets,
                 Clusters<Report>& clusters) {
        using Reports = std::vector<Report*>;
        Targets c;
        clusters.clear();
        ConnectedComponents<Report> cc;
        std::map<Report*, Targets> matching_targets;
        std::map<Target*, Reports> matching_reports;
        for (auto& r : reports) {
            matching_targets[&r] = targettree.query(r.aabbox);
            for (auto t : matching_targets[&r]) {
                matching_reports[t].push_back(&r);
            }
        }

        for (auto& r : reports) {
            cc.init(&r);
            for (auto t : matching_targets[&r]) {
                cc.connect(&r, matching_reports[t]);
            }
        }

        Reports cluster_reports;
        unsigned cluster_id = 1;
        while (cc.get_component(cluster_reports)) {
            Targets cluster_targets;

            // Find all targets
            for (const auto r : cluster_reports) {
                cluster_targets.insert(cluster_targets.end(),
                                       matching_targets[r].begin(),
                                       matching_targets[r].end());
            }

            // Make target list unique
            std::sort(cluster_targets.begin(), cluster_targets.end());
            cluster_targets.erase(std::unique(cluster_targets.begin(),
                                              cluster_targets.end()),
                                  cluster_targets.end());

            // All targets except those in a cluster should later
            // be placed in individual clusters
            std::sort(all_targets.begin(), all_targets.end());
            std::set_difference(all_targets.begin(), all_targets.end(),
                                cluster_targets.begin(), cluster_targets.end(),
                                std::inserter(c, c.begin()));
            all_targets.swap(c);
            c.clear();

            // Cluster id's for post-processing
            for (auto t : cluster_targets) {
                t->cluster = cluster_id;
            }

            // Add cluster to return list
            clusters.emplace_back(cluster_reports, cluster_targets);
            ++cluster_id;
        }

        // All targets not in a cluster should be placed in individual clusters
        for (auto t : all_targets) {
            t->cluster = cluster_id;
            clusters.emplace_back(Reports(), Targets({t}));
            ++cluster_id;
        }
    }

    template<typename Report, typename Sensor>
    void birth(const std::vector<Report>& reports, Sensor& sensor, double time) { // NOLINT
        double rBsum = std::accumulate(
            std::begin(reports),
            std::end(reports),
            0.0,
            [](double s, const Report& r) { return s + r.rB; });

        if (rBsum >= params.r_lim) {
            PARFOR
            for (auto r = std::begin(reports); r < std::end(reports); ++r) {
                double nr = r->rB * sensor.lambdaB / rBsum;
                if (nr >= params.r_lim) {
                    targettree.new_target(std::min(nr, params.rB_max),
                                          sensor.pdf(&params, *r),
                                          time);
                }
            }
        }
    }

    template<typename Report, typename Sensor>
    void cluster_correct(Cluster<Report>& c, const Sensor& sensor) {
        unsigned M = c.targets.size();
        if (M == 0) {
            for (auto& r : c.reports) {
                r->rB = 1;
            }
            return;
        }
        unsigned N = c.reports.size();

        Eigen::MatrixXd C(M, N + 2 * M);
        C.setConstant(inf);
        for (unsigned i = 0; i < M; ++i) {
            c.targets[i]->match(c.reports,
                                sensor,
                                C.block(i, 0, 1, N),
                                C(i, N + i));
            C(i, N + M + i) = c.targets[i]->false_target();
        }

        Murty murty(C);
        Assignment res;
        double cost;
        unsigned n = 0;
        double w;
        double w_sum = 0;
        Eigen::MatrixXd R(M, N + 1);
        R.setZero();
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
            if (w / w_sum < params.w_lim || n >= params.nhyp_max) {
                break;
            }
        }

        if (n == 0) {
            for (auto& r : c.reports) {
                r->rB = 1;
            }
            return;
        }

        R /= w_sum;

        targettree.lock();
        for (unsigned i = 0; i < M; ++i) {
            targettree.remove(c.targets[i]);
        }
        targettree.unlock();

        for (unsigned i = 0; i < M; ++i) {
            c.targets[i]->correct(R.row(i));
        }

        targettree.lock();
        for (unsigned i = 0; i < M; ++i) {
            targettree.replace(c.targets[i]);
        }
        targettree.unlock();

        auto rB = (1.0 - R.colwise().sum().array()).eval();
        for (unsigned j = 0; j < N; ++j) {
            c.reports[j]->rB = rB[j];
        }
    }
};

template<typename PDF>
auto& operator<<(std::ostream& os, const SILMB<PDF>& f) {
    f.repr(os);
    return os;
}

template<typename T>
std::string print(const T& o) {
    std::stringstream s;
    o.repr(s);
    return s.str();
}

}  // namespace lmb

