#pragma once
#include <Eigen/Core>
#include "params.hpp"

namespace lmb {
    template<typename Target>
    struct CV {
        double pS = 1.0;
        double q = 1.0;
        double tscale = 1.0;

        explicit CV(double pS_ = 1.0, double q_ = 1.0)
        : pS(pS_), q(q_) {}

        void operator()(Target* const t, const double time, const double last_time) {
            double dT = (time - last_time) / t->params->tscale;

            Eigen::Matrix4d F;
            F << 1, 0, dT, 0,
                 0, 1, 0, dT,
                 0, 0, 1, 0,
                 0, 0, 0, 1;

            Eigen::Matrix4d Q;
            //double dT2 = dT * dT;
            //double dT3 = dT * dT * dT / 2;
            //double dT4 = dT * dT * dT * dT / 4;
            //Q << dT4, 0,   dT3, 0,
                 //0,   dT4, 0,   dT3,
                 //dT3, 0,   dT2, 0,
                 //0,   dT3, 0,   dT2;
            double dT2 = dT * dT / 2;
            double dT3 = dT * dT * dT / 3;
            Q << dT3, 0,   dT2, 0,
                 0,   dT3, 0,   dT2,
                 dT2, 0,   dT,  0,
                 0,   dT2, 0,   dT;
            Q *= q;

            //std::cout << "Before: " << t->r << ", " << t->pdf.poscov().norm() << ", " << t->pdf.velcov().norm() << std::endl;
            t->pdf.linear_update(F, Q);
            // NOTE: Power, since dT is varying
            t->r *= std::pow(pS, dT);
            //std::cout << "After: " << t->r << ", " << t->pdf.poscov().norm() << ", " << t->pdf.velcov().norm() << std::endl;
        }

        template<typename PDF>
        Eigen::Vector4d predict(const PDF& pdf, const double dT) {
            Eigen::Matrix4d F;
            F << 1, 0, dT, 0,
                 0, 1, 0, dT,
                 0, 0, 1, 0,
                 0, 0, 0, 1;
            return F * pdf.mean().template head<4>();
        }
    };

    template<typename Target>
    struct NM {
        double pS = 1.0;
        Eigen::Matrix4d Q;

        NM() { Q.setZero(); }
        NM(const double pS_, const Eigen::Matrix4d Q_) : pS(pS_), Q(Q_) {}

        void operator()(Target* const t, const double, const double) {
            Eigen::Matrix4d F;
            F.setIdentity();
            t->pdf.linear_update(F, Q);
            t->r *= pS;
        }
    };
}
