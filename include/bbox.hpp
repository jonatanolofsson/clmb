// Copyright 2018 Jonatan Olofsson
#pragma once
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <vector>
#include <cmath>
#include "cf.hpp"
#include "constants.hpp"

namespace lmb {
struct AABBox {
    double min[2];
    double max[2];

    AABBox()
    : min{-inf, -inf},
      max{inf, inf}
    {}

    AABBox(const AABBox& obj) {
        memcpy(min, obj.min, sizeof(min));
        memcpy(max, obj.max, sizeof(max));
    }

    explicit AABBox(const Eigen::Matrix<double, 2, Eigen::Dynamic>& points) {
        auto pmin = points.rowwise().minCoeff();
        min[0] = pmin[0];
        min[1] = pmin[1];
        auto pmax = points.rowwise().maxCoeff();
        max[0] = pmax[0];
        max[1] = pmax[1];
    }

    void operator=(const AABBox& obj) {
        memcpy(min, obj.min, sizeof(min));
        memcpy(max, obj.max, sizeof(max));
    }

    AABBox(double minX, double minY, double maxX, double maxY)
    : min{minX, minY},
      max{maxX, maxY}
    {}

    template<typename T1, typename T2>
    AABBox(const T1& x, const T2& P, double nstd = 2.0) {
        from_gaussian(x, P, nstd);
    }

    explicit AABBox(const Eigen::Matrix2d& P, double nstd = 2.0) {
        from_gaussian(Eigen::Vector2d::Zero(), P, nstd);
    }

    template<typename T1, typename T2>
    void from_gaussian(const T1& x, const T2& P, double nstd = 2.0) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(P);
        auto l = solver.eigenvalues();
        auto e = solver.eigenvectors();

        double r1 = nstd * std::sqrt(l[0]);
        double r2 = nstd * std::sqrt(l[1]);
        double theta = std::atan2(e.col(1).y(), e.col(1).x());

        double ux = r1 * std::cos(theta);
        double uy = r1 * std::sin(theta);
        double vx = r2 * std::cos(theta + M_PI_2);
        double vy = r2 * std::sin(theta + M_PI_2);

        auto dx = std::sqrt(ux*ux + vx*vx);
        auto dy = std::sqrt(uy*uy + vy*vy);

        min[0] = x[0] - dx;
        min[1] = x[1] - dy;
        max[0] = x[0] + dx;
        max[1] = x[1] + dy;
    }

    bool intersects(const AABBox& b) const {
        //     !(b.left   > right  || b.right  < left   || b.top    < bottom || b.bottom > top)      // NOLINT
        return !(b.min[0] > max[0] || b.max[0] < min[0] || b.max[1] < min[1] || b.min[1] > max[1]);  // NOLINT
    }

    template<typename State>
    bool within(const State& x) const {
        return !(x[0] < min[0] || x[0] > max[0] || x[1] < min[1] || x[1] > max[1]);                  // NOLINT
    }

    void repr(std::ostream& o) const {
        o << "AABBox[[" << min[0] << ", " << max[0] << "], [" << min[1] << ", " << max[1] << "]]";       // NOLINT
    }

    AABBox llaabbox(const cf::LL& origin) const {
        auto cmin = cf::ne2ll((Eigen::Vector2d() << min[0], min[1]).finished(), origin);  // NOLINT
        auto cmax = cf::ne2ll((Eigen::Vector2d() << max[0], max[1]).finished(), origin);  // NOLINT
        // if (cmin[1] > cmax[1]) { cmax[1] += MAX_LONGITUDE; }  // FIXME: Lots of special cases!
        return AABBox(cmin[0], cmin[1], cmax[0], cmax[1]);
    }
};

auto& operator<<(std::ostream& os, const AABBox& aabbox) {
    aabbox.repr(os);
    return os;
}

struct BBox {
    using Corners = Eigen::Matrix<double, 2, 4>;
    Corners corners;

    explicit BBox(const Corners& c)
    : corners(c)
    {}

    BBox()
    : corners((Corners() << -inf, -inf, inf, inf,
                            inf, -inf, -inf, inf).finished())
    {}

    template<typename State, typename Covariance>
    BBox(const State& x, const Covariance& P, double nstd = 2.0) {
        from_gaussian(x, P, nstd);
    }

    template<typename Covariance>
    explicit BBox(const Covariance& P, double nstd = 2.0) {
        from_gaussian(Eigen::Vector2d::Zero(), P, nstd);
    }

    BBox llbbox(const cf::LL& origin) const {
        BBox bbox;
        bbox.corners <<
            cf::ne2ll(corners.col(0), origin),
            cf::ne2ll(corners.col(1), origin),
            cf::ne2ll(corners.col(2), origin),
            cf::ne2ll(corners.col(3), origin);
        return bbox;
    }

    BBox nebbox(const cf::LL& origin) const {
        BBox bbox;
        bbox.corners <<
            cf::ll2ne(corners.col(0), origin),
            cf::ll2ne(corners.col(1), origin),
            cf::ll2ne(corners.col(2), origin),
            cf::ll2ne(corners.col(3), origin);
        return bbox;
    }

    template<typename State, typename Covariance>
    void from_gaussian(const State& x, const Covariance& P, double nstd = 2.0) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(P);
        auto l = solver.eigenvalues();
        auto e = solver.eigenvectors();
        double r1 = nstd * std::sqrt(l[0]);
        double r2 = nstd * std::sqrt(l[1]);

        corners << x + r1 * e.col(0) + r2 * e.col(1),
                   x - r1 * e.col(0) + r2 * e.col(1),
                   x - r1 * e.col(0) - r2 * e.col(1),
                   x + r1 * e.col(0) - r2 * e.col(1);
    }

    bool intersects(const BBox& bbox) const {
        for (int i = 0; i < corners.cols(); ++i) {
            if (bbox.within(corners.col(i))) {
                return true;
            }
        }
        for (int i = 0; i < bbox.corners.cols(); ++i) {
            if (within(bbox.corners.col(i))) {
                return true;
            }
        }
        // FIXME: "SWISS CROSS" will fails!
        return false;
    }

    bool intersects(const AABBox& aabb) const {
        return aabbox().intersects(aabb);
    }

    template<typename State>
    bool within(const State& x) const {
        AABBox aabbox_ = aabbox();
        if (!aabbox_.within(x)) {  // Quick check
            return false;
        }
        // https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html
        bool c = false;
        static const int nvert = 4;
        for (int i = 0, j = nvert-1; i < nvert; j = i++) {
            auto& ci = corners.col(i);
            auto& cj = corners.col(j);
            if (((ci.y() > x.y()) != (cj.y() > x.y())) &&
               (x.x() < (cj.x()-ci.x()) * (x.y()-ci.y())
                         / (cj.y() - ci.y()) + ci.x())) {
                 c = !c;
            }
        }
        return c;
    }

    AABBox aabbox() const {
        return AABBox(
            corners.row(0).minCoeff(),
            corners.row(1).minCoeff(),
            corners.row(0).maxCoeff(),
            corners.row(1).maxCoeff());
    }

    void repr(std::ostream& os) const {
        os << "{\"type\":\"BBox\",\"c\":" << corners.format(eigenformat) << "}";
    }
};

auto& operator<<(std::ostream& os, const BBox& bbox) {
    bbox.repr(os);
    return os;
}

}  // namespace lmb
