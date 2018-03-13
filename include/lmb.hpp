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

//#define DEBUG_OUTPUT

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
    unsigned nof_clusters;

    SILMB() {}

    explicit SILMB(const Params& params_) : params(params_) {}


    template<typename Model>
    void predict(Model& model, const double time, const double last_time) {
#ifdef DEBUG_OUTPUT
        std::cout << "\n\nPredict::::::::::" << std::endl;
#endif
        targettree.lock();
        for (auto& t : targettree.targets) {
            targettree.remove(t);
        }
        targettree.unlock();

        PARFOR
        for (auto t = std::begin(targettree.targets);
                t < std::end(targettree.targets); ++t) {
#ifdef DEBUG_OUTPUT
            std::cout << "P: Pre-local: " << **t << std::endl;
#endif
            (*t)->transform_to_local();
#ifdef DEBUG_OUTPUT
            std::cout << "P: Post-local: " << **t << std::endl;
#endif
            (*t)->template predict<Model>(model, time, last_time);
#ifdef DEBUG_OUTPUT
            std::cout << "P: Pre-global: " << **t << std::endl;
#endif
            (*t)->transform_to_global();
#ifdef DEBUG_OUTPUT
            std::cout << "P: Post-global: " << **t << std::endl;
#endif
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
    void correct(const Sensor& sensor, typename Sensor::Scan& scan, double time) {  // NOLINT
        using Report = typename Sensor::Report;
        using Cluster = Cluster_<Report, Target>;
        using Clusters = typename std::vector<Cluster>;
#ifdef DEBUG_OUTPUT
        std::cout << "\n\nCorrect::::::::::" << std::endl;

        std::cout << "FOV: " << sensor.fov.aabbox() << std::endl;
#endif
        auto affected_targets = sensor.get_targets(targettree);
#ifdef DEBUG_OUTPUT
        std::cout << "Affected targets: " << affected_targets.size() << std::endl;
        for (auto& t : affected_targets) {
            std::cout << "\t" << *t << ": " << t->llaabbox() << std::endl;
        }
        std::cout << "All targets: " << std::endl;
        for (auto& t : targettree.targets) {
            std::cout << "\t" << *t << ": " << t->llaabbox() << std::endl;
        }
#endif

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
                             t->id,
                             t->cluster_id);
        }
        targettree.unlock();
        return res;
    }

    TargetSummaries get_targets(const AABBox& aabbox) {
        TargetSummaries res;
        targettree.lock();
        for (auto& t : targettree.query(aabbox)) {
            res.emplace_back(t->pdf.mean(),
                             t->pdf.cov(),
                             t->r,
                             t->id,
                             t->cluster_id);
        }
        targettree.unlock();
        return res;
    }

    Eigen::Array<double, 1, Eigen::Dynamic> pos_phd(const Eigen::Array<double, 2, Eigen::Dynamic>& points) {
        Eigen::Array<double, 2, Eigen::Dynamic> nepoints(2, points.size());
        Eigen::Array<double, 1, Eigen::Dynamic> res(1, points.size());
        res.setZero();
        if (points.size() == 0) {
            return res;
        }

        targettree.lock();

        // Load all affected targets
        AABBox aabbox; aabbox.from_points(points);
        Targets targets = targettree.query(aabbox);
        if (targets.size() == 0) {
            targettree.unlock();
            return res;
        }

        // Transform to local coordinates
        cf::LL origin; origin.setZero();
        for (auto t : targets) { origin += t->pos(); }
        origin /= targets.size();

        for (auto t : targets) { t->transform_to_local(origin); }

        for (unsigned i = 0; i != points.cols(); ++i) {
            nepoints.col(i) = cf::ll2ne(points.col(i), origin);
        }

        // for all points, sum sample target phds
        for (auto t = targets.begin(); t != targets.end(); ++t) {
            (*t)->pos_phd(nepoints, res);
        }

        // Transform back
        for (auto t : targets) { t->transform_to_global(); }

        targettree.unlock();

        return res;
    }

    double ospa(const TargetStates& truth, const double c, const double p) {
        cf::LL origin; origin.setZero();
        for (auto t : truth) { origin += t.template head<2>(); }
        for (auto t : targettree.targets) { origin += t->pos(); }
        origin /= truth.size() + targettree.targets.size();
        return ospa(truth, origin, c, p);
    }

    double ospa(const TargetStates& truth, const cf::LL& origin, const double c, const double p) {
        targettree.lock();
        TargetStates netruth(truth);
        for (unsigned i = 0; i != truth.size(); ++i) {
            cf::ll2ne_i(netruth[i], origin);
        }
        int M = targettree.targets.size();
        int N = netruth.size();
        int n = std::max(M, N);

        if (n == 0) {
            return 0;
        }

        Eigen::MatrixXd C(n, n);
        C.setConstant(c);
        for (int i = 0; i < M; ++i) {
            targettree.targets[i]->transform_to_local(origin);
            targettree.targets[i]->distance(netruth, C.block(i, 0, 1, N));
            targettree.targets[i]->transform_to_global();
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
        cf::LL origin; origin.setZero();
        for (auto t : truth) { origin += t.template head<2>(); }
        for (auto t : targettree.targets) { origin += t->pos(); }
        origin /= truth.size() + targettree.targets.size();
        return gospa(truth, origin, c, p);
    }

    double gospa(const TargetStates& truth, const cf::LL& origin, const double c, const double p) {
        TargetStates netruth(truth);
        for (unsigned i = 0; i != truth.size(); ++i) {
            cf::ll2ne_i(netruth[i], origin);
        }
        targettree.lock();
        int M = targettree.targets.size();
        int N = netruth.size();
        int n = std::max(M, N);

        if (n == 0) {
            return 0;
        }

        double cp = std::pow(c, p);

        Eigen::MatrixXd C(n, n);
        C.setConstant(cp / 2);
        for (int i = 0; i < M; ++i) {
            targettree.targets[i]->transform_to_local(origin);
            targettree.targets[i]->distance(netruth, C.block(i, 0, 1, N));
            targettree.targets[i]->transform_to_global();
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
#ifdef DEBUG_OUTPUT
        std::cout << "Clustering reports: " << scan << std::endl;
        std::cout << "Clustering targets: " << std::endl;
        for (auto& t : all_targets) {
            std::cout << "\t" << t << ": " << *t << std::endl;
        }
#endif
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
#ifdef DEBUG_OUTPUT
                std::cout << "Report matched target: " << *r << " <-> " << **t << std::endl;
#endif
            }
        }

        // Connect targets related by ambiguous reports
        for (auto r = std::begin(scan); r != std::end(scan); ++r) {
            cc.init(&(*r));
            for (auto t = std::begin(matching_targets[&(*r)]); t != std::end(matching_targets[&(*r)]); ++t) {  // NOLINT
                cc.connect(&(*r), matching_reports[*t]);
#ifdef DEBUG_OUTPUT
                std::cout << "Connecting: " << *r << " <-> " << **t << std::endl;
#endif
            }
        }

        Reports cluster_reports;
        unsigned cluster_id = 1;
        while (cc.get_component(cluster_reports)) {
#ifdef DEBUG_OUTPUT
            std::cout << "Cluster: " << std::endl;
            std::cout << "  Reports:" << std::endl;
            for (auto& r : cluster_reports) {
                std::cout << "\t" << *r << std::endl;
            }
#endif
            Targets cluster_targets;

            // Find all targets
            for (const auto r : cluster_reports) {
                cluster_targets.insert(cluster_targets.end(),
                                       matching_targets[r].begin(),
                                       matching_targets[r].end());
                r->cluster_id = cluster_id;
            }

            // Make target list unique
            std::sort(cluster_targets.begin(), cluster_targets.end());
            cluster_targets.erase(std::unique(cluster_targets.begin(),
                                              cluster_targets.end()),
                                  cluster_targets.end());
#ifdef DEBUG_OUTPUT
            std::cout << "  Targets:" << std::endl;
            for (auto& t : cluster_targets) {
                std::cout << "\t" << *t << std::endl;
            }
#endif

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
#ifdef DEBUG_OUTPUT
            std::cout << "Cluster id: " << cluster_id << std::endl;
#endif
        }

        // All targets not in a cluster should be placed in individual clusters
        for (auto t = std::begin(all_targets); t != std::end(all_targets); ++t) {  // NOLINT
            clusters.emplace_back(cluster_id, Reports(), Targets({*t}));
            (*t)->cluster_id = cluster_id;
            ++cluster_id;
        }

        nof_clusters = cluster_id - 1;
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
                                              sensor.pdf(&params, **r, time));
                    }
                }
            }
        }
    }

    template<typename Sensor, typename Cluster>
    void cluster_correct(Cluster& c, const Sensor& sensor) {
        unsigned M = c.targets.size();
        unsigned N = c.reports.size();
        unsigned n = M; // If M > 0, this is the number of generated hypotheses

        // Remove from tree while updating
        targettree.lock();
        for (unsigned i = 0; i < M; ++i) {
            targettree.remove(c.targets[i]);
        }
        targettree.unlock();

        // Transform to NED coordinates
#ifdef DEBUG_OUTPUT
        std::cout << "Pre-local: " << c << std::endl;
#endif
        c.transform_to_local();
#ifdef DEBUG_OUTPUT
        std::cout << "Post-local: " << c << std::endl;
#endif

        if (M > 0) {
            // There are targets to match with
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

#ifdef DEBUG_OUTPUT
            std::cout << "C: \n" << C << std::endl;
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
                if (w / w_sum < params.w_lim || n >= params.nhyp_max) {
                    break;
                }
            }
            if (n > 0) {
                R /= w_sum;

                // Update each target according to the hypotheses' associations
                for (unsigned i = 0; i < M; ++i) {
                    c.targets[i]->correct(R.row(i));
                }

                // Forward birth probabilities to birth algorithm
                auto rB = (1.0 - R.colwise().sum().array()).eval();
                for (unsigned j = 0; j < N; ++j) {
                    c.reports[j]->rB = rB[j];
                }
            }
        }

        // Since n is initialized to M, this also catches M = 0
        if (n == 0) {
            // No targets to match with, or no valid hypotheses?
            for (auto& r : c.reports) {
                r->rB = 1;
            }
        }

        // Move back to LL CF again
#ifdef DEBUG_OUTPUT
        std::cout << "Pre-global: " << c << std::endl;
#endif
        c.transform_to_global();
#ifdef DEBUG_OUTPUT
        std::cout << "Post-global: " << c << std::endl;
#endif

        // Replace the still valid targets into tree
        targettree.lock();
        for (unsigned i = 0; i < M; ++i) {
            targettree.replace(c.targets[i]);
        }
        targettree.unlock();
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

