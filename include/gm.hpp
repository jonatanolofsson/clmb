#pragma once
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <cmath>
#include <numeric>
#include <queue>
#include <set>
#include <vector>
#include "constants.hpp"
#include "gaussian.hpp"
#include "omp.hpp"
#include "params.hpp"
#include "sensors.hpp"

namespace lmb {
    template<int S, int MAX_COMPONENTS = 200>
    struct GM {
        static const int STATES = S;
        typedef GM<STATES, MAX_COMPONENTS> Self;
        typedef Eigen::Matrix<double, STATES, 1> State;
        typedef Eigen::Matrix<double, STATES, STATES> Covariance;
        typedef Gaussian<S> Gaussian;
        Params* params;

        std::vector<Gaussian, Eigen::aligned_allocator<Gaussian>> c;
        BBox bbox;
        AABBox aabbox;
        double eta;

        void normalize() {
            double a = 0;
            for (const auto& g : c) { a += g.w; }
            eta = a;
            if (eta > 1e-8) {
                for (auto& g : c) {
                    g.w /= eta;
                }
            } else {
                clear();
            }
            update_bbox();
        }

        void update_bbox() {
            if (c.size() == 0) {
                return;
            }

            bbox = BBox(mean().template topLeftCorner<2, 1>(),
                        cov().template topLeftCorner<2, 2>(),
                        params->nstd);

            aabbox.min[0] = c[0].m(0);
            aabbox.min[1] = c[0].m(1);
            aabbox.max[0] = c[0].m(0);
            aabbox.max[1] = c[0].m(1);

            for (unsigned i = 0; i < c.size(); ++i) {
                AABBox cbox(c[i].m.template topLeftCorner<2, 1>(),
                            c[i].P.template topLeftCorner<2, 2>(),
                            params->nstd);

                aabbox.min[0] = std::min(aabbox.min[0], cbox.min[0]);
                aabbox.min[1] = std::min(aabbox.min[1], cbox.min[1]);
                aabbox.max[0] = std::max(aabbox.max[0], cbox.max[0]);
                aabbox.max[1] = std::max(aabbox.max[1], cbox.max[1]);
            }
        }

        bool intersects(const BBox& fov) {
            return bbox.intersects(fov);
        }

        bool intersects(const AABBox& fov) {
            return aabbox.intersects(fov);
        }

        void clear() {
            c.clear();
            eta = 0;
        }

        GM() {}
        GM(Params* params_) : params(params_) {}

        GM(Params* params_, const State& mean, const Covariance& cov)
        : params(params_),
          c({Gaussian(1.0, mean, cov)})
        {
            update_bbox();
        }

        template<typename Report, typename Sensor>
        double correct(const Report& z, const Sensor& s) {
            PARFOR
            for (unsigned i = 0; i < c.size(); ++i) {
                c[i].correct(z, s);
            }
            normalize();
            return eta;
        }

        template<typename Sensor>
        double missed(const Sensor& s) {
            for (unsigned i = 0; i < c.size(); ++i) {
                c[i].w *= s.pD(c[i]);
            }
            normalize();
            return 1 - eta;
        }

        template<typename FT, typename QT>
        void linear_update(const FT& F, const QT& Q) {
            PARFOR
            for (unsigned i = 0; i < c.size(); ++i) {
                c[i].m = F * c[i].m;
                c[i].P = F * c[i].P * F.transpose() + Q;
            }
        }

        Self operator+(const Self& rhs) const {
            Self res(*this);
            res.c.reserve(res.c.size() + rhs.c.size());
            res.c.insert(res.c.end(), rhs.c.begin(), rhs.c.end());
            res.prune();
            return res;
        }

        Self operator+=(const Self& rhs) {
            c.reserve(c.size() + rhs.c.size());
            c.insert(c.end(), rhs.c.begin(), rhs.c.end());
            return *this;
        }

        void repr(std::ostream& os) const {
            os << "{\"type\":\"GM\",\"c\":[";
            bool first = true;
            for (unsigned n = 0; n < c.size(); ++n) {
                if (!first) { os << ","; } else { first = false; }
                os << c[n];
            }
            os << "]}";
        }

        void prune() {
            c.erase(std::remove_if(c.begin(), c.end(), [this](auto& t) { return t.w < params->cw_lim; }), c.end());
            std::sort(std::rbegin(c), std::rend(c));
            if (c.size() > MAX_COMPONENTS) {
                c.resize(MAX_COMPONENTS);
            }
            if (c.size() > 0) {
                normalize();
            }
        }

        template<typename W>
        static void join(Self& pdf, const W& w, std::vector<Self>& pdfs) {
            pdf.clear();
            std::vector<unsigned> wi(w.cols());
            std::priority_queue<std::pair<double, unsigned>> q;

            for (unsigned j = 0; j < w.cols(); ++j) {
                wi[j] = 0;
                std::sort(std::rbegin(pdfs[j].c), std::rend(pdfs[j].c));
                q.emplace(w(0, j) * pdfs[j].c[0].w, j);
            }

            double wsum = 0;
            for (unsigned n = 0; (n < MAX_COMPONENTS) && !q.empty(); ++n) {
                auto& p = q.top();
                double nw = p.first;
                wsum += nw;
                unsigned j = p.second;
                if (nw / wsum < pdf.params->cw_lim) {
                    break;
                }
                pdf.c.emplace_back(nw, pdfs[j].c[wi[j]].m, pdfs[j].c[wi[j]].P);
                q.pop();
                if (++wi[j] < pdfs[j].c.size()) {
                    q.emplace(w(0, j) * pdfs[j].c[wi[j]].w, j);
                }
            }
            pdf.normalize();
        }

        State mean() const {
            State res;
            res.setZero();
            for (auto& cmp : c) {
                res += cmp.w * cmp.m;
            }
            return res;
        }

        Covariance cov(const State& m) const {
            Covariance res;
            res.setZero();
            for (auto& cmp : c) {
                auto d = cmp.m - m;
                res += cmp.w * (cmp.P + d * d.transpose());
            }
            return res;
        }

        Covariance cov() const {
            return cov(mean());
        }
    };

    template<int STATES, int MAX_COMPONENTS = 200>
    auto& operator<<(std::ostream& os, const GM<STATES, MAX_COMPONENTS>& t) {
        t.repr(os);
        return os;
    }
}
