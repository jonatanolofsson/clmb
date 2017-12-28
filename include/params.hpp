#pragma once

namespace lmb {
    struct Params {
        double w_lim = 0.01;
        double r_lim = 0.05;
        double nhyp_max = 100;
        double rB_max = 0.8;
        double nstd = 1.9;
        double cw_lim = 0.01;
    };
}
