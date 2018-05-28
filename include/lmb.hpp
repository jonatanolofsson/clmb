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
#include "omp.hpp"
#include "params.hpp"
#include "sensors.hpp"
#include "target.hpp"
#include "targettree.hpp"

//#define DEBUG_OUTPUT

namespace lmb {
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
    using Params = Params;


    Params params;

    TargetTree targettree;
    unsigned nof_clusters;
    std::vector<int> cluster_ntargets;
    std::vector<int> cluster_nreports;
    std::vector<int> cluster_nhyps;

    SILMB() : targettree(&params) {}

    explicit SILMB(const Params& params_) : params(params_), targettree(&params) {}


    template<typename Model>
    void predict(Model& model, const double time, const double last_time) {
#ifdef DEBUG_OUTPUT
        std::cout << "\n\nPredict::::::::::" << std::endl;
#endif
        CRITICAL(ttree)
        {
            auto all_targets = targettree.targets;
            targettree.remove_all();

            PARFOR
            for (auto t = std::begin(all_targets);
                    t < std::end(all_targets); ++t) {
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

            for (auto& t : all_targets) {
                if (t->viable()) {
                    targettree.replace(t);
                } else {
                    targettree.erase(t);
                }
            }
        }
    }

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
        CRITICAL(ttree)
        {
            for (auto& t : targettree.targets) {
                std::cout << "\t" << *t << ": " << t->llaabbox() << std::endl;
            }
        }
#endif

        Clusters clusters;
        cluster<Sensor>(scan, affected_targets, clusters);
        cluster_ntargets.resize(clusters.size());
        cluster_nreports.resize(clusters.size());
        cluster_nhyps.resize(clusters.size());

        PARFOR
        for (auto c = std::begin(clusters); c < std::end(clusters); ++c) {
            cluster_ntargets[c->id] = c->targets.size();
            cluster_nreports[c->id] = c->reports.size();
            c->correct(targettree, sensor, time);
            cluster_nhyps[c->id] = c->n;
        }

        // Reports have now moved to their respective clusters' NE system
        birth(clusters, scan, sensor, time);
    }

    double enof_targets() {
        double res = 0;
        CRITICAL(ttree)
        {
            for (auto& t : targettree.targets) {
                res += t->r;
            }
        }
        return res;
    }

    double enof_targets(const AABBox& aabbox) {
        double res = 0;
        CRITICAL(ttree)
        {
            for (auto& t : targettree.query(aabbox)) {
                res += t->r;
            }
        }
        return res;
    }

    unsigned nof_targets(const double r_lim = 0.7) {
        unsigned res = 0;
        CRITICAL(ttree)
        {
            for (auto& t : targettree.targets) {
                if (t->r >= r_lim) { ++res; }
            }
        }
        return res;
    }

    unsigned nof_targets(const AABBox& aabbox, const double r_lim = 0.7) {
        unsigned res = 0;
        CRITICAL(ttree)
        {
            for (auto& t : targettree.query(aabbox)) {
                if (t->r >= r_lim) { ++res; }
            }
        }
        return res;
    }

    TargetSummaries get_targets() {
        TargetSummaries res;
        CRITICAL(ttree)
        {
            for (auto& t : targettree.targets) {
                res.emplace_back(t->pdf.mean(),
                                 t->pdf.cov(),
                                 t->r,
                                 t->id,
                                 t->cid);
            }
        }
        return res;
    }

    TargetSummaries get_targets(const AABBox& aabbox) {
        TargetSummaries res;
        CRITICAL(ttree)
        {
            for (auto& t : targettree.query(aabbox)) {
                res.emplace_back(t->pdf.mean(),
                                 t->pdf.cov(),
                                 t->r,
                                 t->id,
                                 t->cid);
            }
        }
        return res;
    }

    Eigen::Array<double, 1, Eigen::Dynamic> pos_phd(const Eigen::Array<double, 2, Eigen::Dynamic>& points, const Eigen::Vector2d& gridsize) {
        Eigen::Array<double, 2, Eigen::Dynamic> nepoints(2, points.cols());
        Eigen::Array<double, 1, Eigen::Dynamic> res(1, points.cols());
        res.setZero();
        if (points.cols() == 0) { return res; }

        AABBox neaabbox(0.0, 0.0, gridsize.x(), gridsize.y());
        for (unsigned p = 0; p != points.cols(); ++p) {
            auto point2 = cf::ne2ll(gridsize, points.col(p));
            AABBox llaabbox(points(0, p), points(1, p), point2(0), point2(1));
            //std::cout << "phd aabbbox: " << points.col(p).transpose() << " -> " << point2.transpose() << " :: " << aabbox << std::endl;
            Targets targets = targettree.query(llaabbox);
            if (targets.size() == 0) { continue; }
            PARFOR
            for (std::size_t t = 0; t < targets.size(); ++t) { targets[t]->transform_to_local(points.col(p)); }
            double phdval = 0;
            OMP(parallel for reduction(+:phdval))
            for (std::size_t t = 0; t < targets.size(); ++t) {
                phdval += targets[t]->r * targets[t]->pdf.overlap(neaabbox);
            }
            res[p] = phdval;
            PARFOR
            for (std::size_t t = 0; t < targets.size(); ++t) { targets[t]->transform_to_global(); }
        }

        return res;
    }

    double ospa(const TargetStates& truth, const double c, const double p) {
        cf::LL origin; origin.setZero();
        for (auto t : truth) { origin += t.template head<2>(); }
        CRITICAL(ttree)
        {
            for (auto t : targettree.targets) { origin += t->pos(); }
            origin /= truth.size() + targettree.targets.size();
        }
        return ospa(truth, origin, c, p);
    }

    double ospa(const TargetStates& truth, const cf::LL& origin, const double c, const double p) {
        double score = 0;
        TargetStates netruth(truth);
        for (unsigned i = 0; i != truth.size(); ++i) {
            cf::ll2ne_i(netruth[i], origin);
        }
        CRITICAL(ttree)
        {
            int M = targettree.targets.size();
            int N = netruth.size();
            int n = std::max(M, N);

            if (n > 0) {
                Eigen::MatrixXd C(n, n);
                C.setConstant(c);
                for (int i = 0; i < M; ++i) {
                    targettree.targets[i]->transform_to_local(origin);
                    targettree.targets[i]->distance(netruth, C.block(i, 0, 1, N));
                    targettree.targets[i]->transform_to_global();
                }
                C = C.cwiseMin(c);

                // Note that since exp() is strictly increasing, C and C.^p yields the
                // same assignments. By only raising the result, we compute fewer
                // exponentials
                auto res = lap::lap(C);

                double cost = 0;
                for (unsigned i = 0; i < res.rows(); ++i) {
                    cost += std::pow(C(i, res[i]), p);
                }
                score = std::pow(cost / n, 1.0 / p);
            }
        }
        return score;
    }

    double gospa(const TargetStates& truth, const double c, const double p) {
        cf::LL origin; origin.setZero();
        for (auto t : truth) { origin += t.template head<2>(); }
        CRITICAL(ttree)
        {
            for (auto t : targettree.targets) { origin += t->pos(); }
            origin /= truth.size() + targettree.targets.size();
        }
        return gospa(truth, origin, c, p);
    }

    double gospa(const TargetStates& truth, const cf::LL& origin, const double c, const double p) {
        TargetStates netruth(truth);
        for (unsigned i = 0; i != truth.size(); ++i) {
            cf::ll2ne_i(netruth[i], origin);
        }

        double score = 0;
        CRITICAL(ttree)
        {
            int M = targettree.targets.size();
            int N = netruth.size();
            int n = std::max(M, N);

            if (n > 0) {
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

                auto res = lap::lap(C);
                score = std::pow(lap::cost(C, res), 1.0 / p);
            }
        }
        return score;
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
        unsigned cid = 0;
        while (cc.get_component(cluster_reports)) {
            Targets cluster_targets;

            // Find all targets
            for (const auto r : cluster_reports) {
                cluster_targets.insert(cluster_targets.end(),
                                       matching_targets[r].begin(),
                                       matching_targets[r].end());
                r->cid = cid;
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

            auto cluster_obj = clusters.emplace_back(cid,
                                                     cluster_reports,
                                                     cluster_targets,
                                                     &params);
            // Cluster id's for post-processing
            for (auto t = std::begin(cluster_targets); t != std::end(cluster_targets); ++t) {  // NOLINT
                (*t)->cid = cid;
            }

#ifdef DEBUG_OUTPUT
            std::cout << "Cluster: " << cid << std::endl;
            std::cout << "  Reports:" << std::endl;
            for (auto& r : cluster_reports) {
                std::cout << "\t" << *r << std::endl;
            }
            std::cout << "  Targets:" << std::endl;
            for (auto& t : cluster_targets) {
                std::cout << "\t" << *t << std::endl;
            }
#endif

            ++cid;
        }

        // All targets not in a cluster should be placed in individual clusters
        for (auto t = std::begin(all_targets); t != std::end(all_targets); ++t) {  // NOLINT
            clusters.emplace_back(cid, Reports(), Targets({*t}), &params);
            (*t)->cid = cid;
            ++cid;
        }

        nof_clusters = cid;
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
            for (auto c = std::begin(clusters); c < std::end(clusters); ++c) {
                PARFOR
                for (auto r = std::begin(c->reports); r < std::end(c->reports); ++r) {                    // NOLINT
                    double nr = (*r)->rB * sensor.lambdaB / rBsum;
                    if (nr >= params.r_lim) {
                        CRITICAL(ttree)
                        {
                            auto t = targettree.new_target(
                                std::min(nr, params.rB_max),
                                sensor.pdf_init1(&params, **r, time), time);
                            t->cid = (*r)->cid;
                        }
                    }
                }
            }
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

