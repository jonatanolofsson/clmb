#pragma once
#include <Eigen/Core>

namespace lmb {
    static double NSTD = 2.0;
    static double inf = 30000;
    static Eigen::IOFormat eigenformat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",", "", "", "[", "]");
}
