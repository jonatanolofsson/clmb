#pragma once
#include <cmath>
#include <set>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "murty.hpp"
#include "sensors.hpp"
#include "target.hpp"
#include "bbox.hpp"
#include "connectedcomponents.hpp"
#include "targettree.hpp"

namespace lmb {
    template<typename PDF>
    struct SILMB {
        typedef lmb::Target<PDF> Target;
        typedef lmb::Targets<PDF> Targets;

        double w_lim;
        double r_lim;
        double nhyp_max;
        double rB_max;
        double lambdaB;

        TargetTree<PDF> targets;

        SILMB()
        : w_lim(0.01),
          r_lim(0.01),
          nhyp_max(100),
          rB_max(0.5),
          lambdaB(1)
        {}

        template<void(*model)(Target* const, double)>
        void predict(double time) {
            targets.lock();
            for (auto& t : targets.targets) {
                targets.remove(t);
            }
            targets.unlock();

            #pragma omp parallel for
            //for (auto& t : targets.targets) {
            for (auto t = std::begin(targets.targets); t != std::end(targets.targets); ++t) {
                *t->template predict<model>(time);
            }

            targets.lock();
            for (auto& t : targets.targets) {
                targets.replace(t);
            }
            targets.unlock();
        }

        template<void(*model)(Target* const, double)>
        void predict(const AABBox& aabbox, double time) {
            Targets all_targets;
            targets.query(aabbox, all_targets);

            targets.lock();
            for (auto& t : all_targets) {
                targets.remove(t);
            }
            targets.unlock();

            #pragma omp parallel for
            //for (auto& t : all_targets) {
            for (auto t = std::begin(all_targets); t < std::end(all_targets); ++t) {
                (*t)->template predict<model>(time);
            }

            targets.lock();
            for (auto& t : all_targets) {
                targets.replace(t);
            }
            targets.unlock();
        }

        template<typename Report>
        void cluster(std::vector<Report>& reports, Targets& all_targets, Clusters<Report, Target>& clusters) {
            typedef std::vector<Report*> Reports;
            Targets c;
            clusters.clear();
            ConnectedComponents<Report> cc;
            std::map<Report*, Targets> matching_targets;
            std::map<Target*, Reports> matching_reports;
            for (auto& r : reports) {
                targets.query(r.aabbox, matching_targets[&r]);
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
                    cluster_targets.insert(cluster_targets.end(), matching_targets[r].begin(), matching_targets[r].end());
                }

                // Make target list unique
                std::sort(cluster_targets.begin(), cluster_targets.end());
                cluster_targets.erase(std::unique(cluster_targets.begin(), cluster_targets.end()), cluster_targets.end());

                // All targets except those in a cluster should later be placed in individual clusters
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
        void correct(std::vector<Report>& reports, const Sensor& sensor, double time) {
            Clusters<Report, Target> clusters;
            Targets all_targets;
            targets.query(sensor.aabbox, all_targets);
            //sensor.fov_filter(all_targets); // FIXME

            cluster(reports, all_targets, clusters);

            #pragma omp parallel for
            //for (auto& c : clusters) {
            for (auto c = std::begin(clusters); c < std::end(clusters); ++c) {
                cluster_correct<Report, Sensor>(*c, sensor);
            }

            birth(reports, sensor, time);
        }

        template<typename Reports, typename Sensor>
        void birth(const Reports& reports, Sensor& sensor, double time) {
            double rBsum = std::accumulate(std::begin(reports), std::end(reports), 0.0, [](double s, const Report& r) { return s + r.rB; });
            //std::cout << "rBsum: " << std::accumulate(std::begin(reports), std::end(reports), 0.0, [](double s, const Report& r) { return s + r.rB; }) << std::endl;

            if (rBsum >= r_lim) {
                //#pragma omp parallel for
                //for (auto& r : reports) {
                for (auto r = std::begin(reports); r < std::end(reports); ++r) {
                    double nr = r->rB * lambdaB / rBsum;
                    //std::cout << "nr: " << nr << std::endl;
                    //std::cout << "r_lim: " << r_lim << std::endl;
                    if (nr >= r_lim) {
                        targets.new_target(std::min(nr, rB_max), sensor.pdf(*r), time);
                    }
                }
            }
        }

        template<typename Report, typename Sensor>
        void cluster_correct(Cluster<Report, Target>& c, const Sensor& sensor) {
            unsigned N = c.targets.size();
            if (N == 0) {
                for (auto& r : c.reports) {
                    r->rB = 1;
                }
                return;
            }
            unsigned M = c.reports.size();

            Eigen::MatrixXd C(N, M + 2 * N);
            C.setConstant(inf);
            for (unsigned i = 0; i < N; ++i) {
                c.targets[i]->match(c.reports, sensor, C.block(i, 0, 1, M), C(i, M + i));
                C(i, M + N + i) = c.targets[i]->false_target();
            }

            Murty murty(C);
            Assignment res;
            double cost;
            unsigned n = 0;
            double w;
            double w_sum = 0;
            Eigen::MatrixXd R(N, M + 1);
            R.setZero();
            while(murty.draw(res, cost)) {
                w = std::exp(-cost);
                w_sum += w;
                for (unsigned i = 0; i < N; ++i) {
                    if ((unsigned)res[i] < M) {
                        R(i, res[i]) += w;
                    } else if ((unsigned)res[i] == M + i) {
                        R(i, M) += w;
                    }
                }
                ++n;
                if (w / w_sum < w_lim or n >= nhyp_max) {
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
            //std::cout << "R: " << std::endl << R << std::endl;

            targets.lock();
            for (unsigned i = 0; i < N; ++i) {
                targets.remove(c.targets[i]);
            }
            targets.unlock();

            for (unsigned i = 0; i < N; ++i) {
                c.targets[i]->correct(R.row(i));
            }

            targets.lock();
            for (unsigned i = 0; i < N; ++i) {
                targets.replace(c.targets[i]);
            }
            targets.unlock();

            auto rB = (1.0 - R.colwise().sum().array()).eval();
            for (unsigned j = 0; j < M; ++j) {
                c.reports[j]->rB = rB[j];
            }
        }

        void dump(std::ostream& os) const {
            os << "{\"targets\":[";
            bool first = true;
            for (auto& t : targets.targets) {
                if (!first) { os << ","; } else { first = false; }
                os << *t;
            }
            os << "]}";
        }
    };

    template<typename PDF>
    auto& operator<<(std::ostream& os, const SILMB<PDF>& f) {
        f.dump(os);
        return os;
    }
}
