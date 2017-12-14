#pragma once
#include <vector>
#include <numeric>
#include <Eigen/Dense>
#include <cmath>
#include <set>
#include <queue>
#include "sensors.hpp"
#include "constants.hpp"
#include <Eigen/StdVector>

namespace lmb {
    static const double gmw_lim = 0.05; // FIXME

    template<int S, int MAX_COMPONENTS = 200> // FIXME: Enforce MAX_COMPONENTS
    struct GM {
        static const int STATES = S;
        typedef GM<STATES, MAX_COMPONENTS> Self;
        typedef Eigen::Matrix<double, STATES, 1> State;
        typedef Eigen::Matrix<double, STATES, STATES> Covariance;

        struct alignas(16) GaussianComponent {
            State m;
            Covariance P;
            double w;

            GaussianComponent(double w_, const State& m_, const Covariance& P_)
            : m(m_),
              P(P_),
              w(w_)
            {}

            bool operator<(const GaussianComponent& b) const {
                return w < b.w;
            }
        };

        std::vector<GaussianComponent, Eigen::aligned_allocator<GaussianComponent>> c;
        AABBox aabbox;
        double pD;
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

            aabbox.min[0] = c[0].m(0);
            aabbox.min[1] = c[0].m(1);
            aabbox.max[0] = c[0].m(0);
            aabbox.max[1] = c[0].m(1);

            for (unsigned i = 0; i < c.size(); ++i) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(c[i].P.template topLeftCorner<2, 2>());
                auto l = solver.eigenvalues();
                auto e = solver.eigenvectors();

                double r1 = NSTD * std::sqrt(l[0]);
                double r2 = NSTD * std::sqrt(l[1]);
                double theta = std::atan2(e.col(1).y(), e.col(1).x());

                double ux = r1 * std::cos(theta);
                double uy = r1 * std::sin(theta);
                double vx = r2 * std::cos(theta + M_PI_2);
                double vy = r2 * std::sin(theta + M_PI_2);

                auto dx = std::sqrt(ux*ux + vx*vx);
                auto dy = std::sqrt(uy*uy + vy*vy);

                aabbox.min[0] = std::min(aabbox.min[0], c[i].m(0) - dx);
                aabbox.min[1] = std::min(aabbox.min[1], c[i].m(1) - dy);
                aabbox.max[0] = std::max(aabbox.max[0], c[i].m(0) + dx);
                aabbox.max[1] = std::max(aabbox.max[1], c[i].m(1) + dy);
            }
        }

        void clear() {
            c.clear();
            eta = 0;
        }

        GM()
        : pD(1.0)
        {}

        GM(const State& mean, const Covariance& cov)
        : c({GaussianComponent(1.0, mean, cov)}),
          aabbox(mean.template topRows<2>(), cov.template topLeftCorner<2, 2>()),
          pD(1.0)
        {}

        template<typename Measurement, typename Sensor>
        double correct(const Measurement& z, const Sensor& s) {
            for (unsigned i = 0; i < c.size(); ++i) {
                auto dz = (z.z - s.measurement(c[i].m)).eval();
                auto Dinv = (s.H * c[i].P * s.H.transpose() + z.R).inverse().eval();
                auto K = c[i].P * s.H.transpose() * Dinv;
                c[i].w *= pD * z.likelihood(dz, Dinv) / z.kappa;
                c[i].m += K * dz;
                c[i].P -= K * s.H * c[i].P;
            }
            normalize();
            return eta;
        }

        template<typename Sensor>
        double missed(const Sensor&) {
            return 1 - pD;
        }

        template<typename FT, typename QT>
        void linear_update(const FT& F, const QT& Q) {
            #pragma omp parallel for
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

        void dump(std::ostream& os) const {
            os << "{\"type\":\"GM\",\"c\":[";
            bool first = true;
            for (unsigned n = 0; n < c.size(); ++n) {
                if (!first) { os << ","; } else { first = false; }
                os << "{\"w\":" << c[n].w << ",\"m\":" << c[n].m.format(eigenformat) << ",\"P\":" << c[n].P.format(eigenformat) << "}";
            }
            os << "]}";
        }

        void prune() {
            c.erase(std::remove_if(c.begin(), c.end(), [](auto& t) { return t.w < gmw_lim; }), c.end());
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
                if (nw / wsum < gmw_lim) {
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
        t.dump(os);
        return os;
    }
}
