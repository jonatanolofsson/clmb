#pragma once
#include <Eigen/Core>

namespace lmb {
    template<unsigned QSCALE=1000, typename Target>
    void CV(Target* const t, double time) {
        double dT = (time - t->t);
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
        Q *= QSCALE / 1000.0;

        for (unsigned i = 0; i < t->pdf.c.size(); ++i) {
            t->pdf.c[i].m = F * t->pdf.c[i].m;
            t->pdf.c[i].P = F * t->pdf.c[i].P * F.transpose() + Q;
        }
        t->t = time;
    }

    template<typename Target>
    void NM(Target* const t, double time) {
        t->t = time;
    }
}
