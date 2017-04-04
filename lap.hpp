#pragma once

#include <Eigen/Core>

namespace lap {
    typedef Eigen::Matrix<int, Eigen::Dynamic, 1> Assignment;
    typedef Eigen::Matrix<int, Eigen::Dynamic, 1> Slack;
    typedef Eigen::MatrixXd CostMatrix;

    void lap(const CostMatrix&, Assignment&, Slack&, Slack&, double&);
}
