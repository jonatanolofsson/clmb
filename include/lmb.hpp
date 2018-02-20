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
#include "cf.hpp"
#include "cluster.hpp"
#include "connectedcomponents.hpp"
#include "lap.hpp"
#include "murty.hpp"
#include "omp.hpp"
#include "params.hpp"
#include "sensors.hpp"
#include "target.hpp"
#include "targettree.hpp"

namespace lmb {

using lap::Murty;
using lap::Assignment;

template<typename PDF_>
class SILMB {
 public:
    using PDF = PDF_;
    using Self = SILMB<PDF>;
    using Target = Target_<PDF>;
    using Targets = Targets_<PDF>;
    using TargetTree = TargetTree_<PDF>;
    using Gaussian = Gaussian_<PDF::STATES>;
    using TargetStates = std::vector<typename PDF::State>;
    using TargetSummaries = TargetSummaries_<PDF::STATES>;


    Params params;

    TargetTree targettree;

    SILMB() {}

    explicit SILMB(const Params& params_) : params(params_) {}


    template<typename Model>
    void predict(Model& model, double time) {
        //std::cout << "\n\nPredict::::::::::" << std::endl;
        targettree.lock();
        for (auto& t : targettree.targets) {
            targettree.remove(t);
        }
        targettree.unlock();

        PARFOR
        for (auto t = std::begin(targettree.targets);
                t < std::end(targettree.targets); ++t) {
            //std::cout << "Pre-local: " << **t << std::endl;
            (*t)->transform_to_local();
            //std::cout << "Post-local: " << **t << std::endl;
            (*t)->template predict<Model>(model, time);
            //std::cout << "Pre-global: " << **t << std::endl;
            (*t)->transform_to_global();
            //std::cout << "Post-global: " << **t << std::endl;
        }

        targettree.lock();
        for (auto& t : targettree.targets) {
            targettree.replace(t);
        }
        targettree.unlock();
    }

    //template<typename Model>
    //void predict(Model& model, const AABBox& aabbox, double time) {
        //auto all_targets = targettree.query(aabbox);

        //targettree.lock();
        //for (auto& t : all_targets) {
            //targettree.remove(t);
        //}
        //targettree.unlock();

        //PARFOR
        //for (auto t = std::begin(all_targets); t < std::end(all_targets); ++t) {
            //(*t)->template predict<Model>(model, time);
        //}

        //targettree.lock();
        //for (auto& t : all_targets) {
            //targettree.replace(t);
        //}
        //targettree.unlock();
    //}

    template<typename Sensor>
    void correct(typename Sensor::Scan& scan, const Sensor& sensor, double time) {  // NOLINT
        using Report = typename Sensor::Report;
        using Cluster = Cluster_<Report, Target>;
        using Clusters = typename std::vector<Cluster>;
        //std::cout << "\n\nCorrect::::::::::" << std::endl;

        //std::cout << "FOV: " << sensor.fov.aabbox() << std::endl;
        auto affected_targets = sensor.get_targets(targettree);
        //std::cout << "Affected targets: " << affected_targets.size() << std::endl;
        //std::cout << "All targets: " << std::endl;
        //for (auto& t : targettree.targets) {
            //std::cout << "\t" << *t << ": " << t->llaabbox() << std::endl;
        //}

        Clusters clusters;
        cluster<Sensor>(scan, affected_targets, clusters);

        PARFOR
        for (auto c = std::begin(clusters); c < std::end(clusters); ++c) {
            cluster_correct<Sensor>(*c, sensor);
        }

        // Reports have now moved to their respective clusters' NE system
        birth(clusters, scan, sensor, time);
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
            res.emplace_back(t->pdf.mean(),
                             t->pdf.cov(),
                             t->r,
                             t->id);
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
    template<typename Sensor, typename Clusters>
    void cluster(typename Sensor::Scan& scan,
                 Targets& all_targets,
                 Clusters& clusters) {
        using Report = typename Sensor::Report;
        using Reports = std::vector<Report*>;
        //std::cout << "Clustering reports: " << scan << std::endl;
        //std::cout << "Clustering targets: " << std::endl;
        //for (auto& t : all_targets) {
            //std::cout << "\t" << t << ": " << *t << std::endl;
        //}
        // Clustering is performed in the LL CF
        Targets c;
        clusters.clear();
        ConnectedComponents<Report> cc;
        std::map<Report*, Targets> matching_targets;
        std::map<Target*, Reports> matching_reports;

        // Match reports to potential target matches
        for (auto r = std::begin(scan); r != std::end(scan); ++r) {
            matching_targets[&(*r)] = targettree.query(r->llaabbox());
            for (auto t = std::begin(matching_targets[&(*r)]); t != std::end(matching_targets[&(*r)]); ++t) {  // NOLINT
                matching_reports[*t].push_back(&(*r));
                //std::cout << "Report matched target: " << *r << " <-> " << **t << std::endl;
            }
        }

        // Connect targets related by ambiguous reports
        for (auto r = std::begin(scan); r != std::end(scan); ++r) {
            cc.init(&(*r));
            for (auto t = std::begin(matching_targets[&(*r)]); t != std::end(matching_targets[&(*r)]); ++t) {  // NOLINT
                cc.connect(&(*r), matching_reports[*t]);
                //std::cout << "Connecting: " << *r << " <-> " << **t << std::endl;
            }
        }

        Reports cluster_reports;
        unsigned cluster_id = 1;
        while (cc.get_component(cluster_reports)) {
            //std::cout << "Cluster: " << std::endl;
            //std::cout << "  Reports:" << std::endl;
            //for (auto& r : cluster_reports) {
                //std::cout << "\t" << *r << std::endl;
            //}
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
            //std::cout << "  Targets:" << std::endl;
            //for (auto& t : cluster_targets) {
                //std::cout << "\t" << *t << std::endl;
            //}

            // All targets except those in a cluster should later
            // be placed in individual clusters
            std::sort(all_targets.begin(), all_targets.end());
            std::set_difference(all_targets.begin(), all_targets.end(),
                                cluster_targets.begin(), cluster_targets.end(),
                                std::inserter(c, c.begin()));
            all_targets.swap(c);
            c.clear();

            auto cluster_obj = clusters.emplace_back(cluster_id,
                                                     cluster_reports,
                                                     cluster_targets);
            // Cluster id's for post-processing
            for (auto t = std::begin(cluster_targets); t != std::end(cluster_targets); ++t) {  // NOLINT
                (*t)->cluster_id = cluster_id;
            }

            ++cluster_id;
            //std::cout << "Cluster id: " << cluster_id << std::endl;
        }

        // All targets not in a cluster should be placed in individual clusters
        for (auto t = std::begin(all_targets); t != std::end(all_targets); ++t) {  // NOLINT
            clusters.emplace_back(cluster_id, Reports(), Targets({*t}));
            (*t)->cluster_id = cluster_id;
            ++cluster_id;
        }
    }

    template<typename Sensor, typename Clusters>
    void birth(const Clusters& clusters, const typename Sensor::Scan& scan, Sensor& sensor, double time) { // NOLINT
        using Report = typename Sensor::Report;
        double rBsum = std::accumulate(
            std::begin(scan),
            std::end(scan),
            0.0,
            [](double s, const Report& r) { return s + r.rB; });

        if (rBsum >= params.r_lim) {
            PARFOR
            for (auto c = std::begin(clusters); c != std::end(clusters); ++c) {
                PARFOR
                for (auto r = std::begin(c->reports); r != std::end(c->reports); ++r) {                    // NOLINT
                    double nr = (*r)->rB * sensor.lambdaB / rBsum;
                    if (nr >= params.r_lim) {
                        targettree.new_target(std::min(nr, params.rB_max),
                                              sensor.pdf(&params, **r),
                                              time);
                    }
                }
            }
        }
    }

    template<typename Sensor, typename Cluster>
    void cluster_correct(Cluster& c, const Sensor& sensor) {
        unsigned M = c.targets.size();
        unsigned N = c.reports.size();

        // Remove from tree while updating
        targettree.lock();
        for (unsigned i = 0; i < M; ++i) {
            targettree.remove(c.targets[i]);
        }
        targettree.unlock();

        // Transform to NED coordinates
        //std::cout << "Pre-local: " << c << std::endl;
        c.transform_to_local();
        //std::cout << "Post-local: " << c << std::endl;

        // No targets to match with? Must be potential new targets then
        if (M == 0) {
            for (auto& r : c.reports) {
                r->rB = 1;
            }
            // Move back to LL CF again
            //std::cout << "Pre-global 0: " << c << std::endl;
            c.transform_to_global();
            //std::cout << "Post-global 0: " << c << std::endl;
            return;
        }

        // Create cost matrix
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
            if (w / w_sum < params.w_lim || n >= params.nhyp_max) {
                break;
            }
        }

        // No valid hypotheses?! Quit this farce!
        if (n == 0) {
            for (auto& r : c.reports) {
                r->rB = 1;
            }
            // Move back to LL CF again
            //std::cout << "Pre-global 1: " << c << std::endl;
            c.transform_to_global();
            //std::cout << "Post-global 1: " << c << std::endl;
            return;
        }

        R /= w_sum;

        // Update each target according to the hypotheses' associations
        for (unsigned i = 0; i < M; ++i) {
            c.targets[i]->correct(R.row(i));
        }

        // Move back to LL CF again
        //std::cout << "Pre-global: " << c << std::endl;
        c.transform_to_global();
        //std::cout << "Post-global: " << c << std::endl;

        // Replace the still valid targets into tree
        targettree.lock();
        for (unsigned i = 0; i < M; ++i) {
            targettree.replace(c.targets[i]);
        }
        targettree.unlock();

        // Forward birth probabilities to birth algorithm
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

