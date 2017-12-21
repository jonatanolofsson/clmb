#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include "constants.hpp"

namespace lmb {
    struct AABBox {
        double min[2];
        double max[2];

        AABBox()
        : min{-inf, -inf},
          max{inf, inf}
        {}

        AABBox(const AABBox& obj)
        : min{obj.min[0], obj.min[1]},
          max{obj.max[0], obj.max[1]}
        {}

        void operator=(const AABBox& obj) {
            min[0] = obj.min[0];
            min[1] = obj.min[1];
            max[0] = obj.max[0];
            max[1] = obj.max[1];
        }

        AABBox(double minX, double minY, double maxX, double maxY)
        : min{minX, minY},
          max{maxX, maxY}
        {}

        template<typename T1, typename T2>
        AABBox(const T1& x, const T2& P, double nstd = 2.0) {
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

        void repr(std::ostream& o) const {
            o << "AABBox[" << min[0] << ", " << min[1] << ", " << max[0] << ", " << max[1] << "]";
        }
    };

    auto& operator<<(std::ostream& os, const AABBox& bbox) {
        bbox.repr(os);
        return os;
    }

    struct BBox {
        typedef Eigen::Matrix<double, 2, 4> Corners;
        Corners corners;

        BBox(const Corners& c)
        : corners(c)
        {}

        BBox()
        : corners((Corners() << -inf, -inf, inf, inf, inf, -inf, -inf, inf).finished())
        {}

        bool intersects(const AABBox&) const {
            // FIXME
            return true;
        }

        AABBox aabbox() {
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
}
