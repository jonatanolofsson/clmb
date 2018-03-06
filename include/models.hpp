#pragma once
#include <Eigen/Core>

namespace lmb {
    template<typename Target>
    struct CV {
        double pS = 1.0;
        double q = 1.0;

        CV() {}
        CV(double pS_, double q_) : pS(pS_), q(q_) {}

        void operator()(Target* const t, const double time, const double last_time) {
            double dT = (time - last_time);
            double dT2 = dT * dT / 2;
            double dT3 = dT * dT * dT / 3;

            Eigen::Matrix4d F;
            F << 1, 0, dT, 0,
                 0, 1, 0, dT,
                 0, 0, 1, 0,
                 0, 0, 0, 1;

            Eigen::Matrix4d Q;
            Q << dT3, 0, dT2, 0,
                 0, dT3, 0, dT2,
                 dT2, 0, dT, 0,
                 0, dT2, 0, dT;
            Q *= q;

            t->pdf.linear_update(F, Q);
            // NOTE: Power, since dT is varying
            t->r *= std::pow(pS, dT / 3600);
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
