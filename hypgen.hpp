#pragma once

#include "lap.hpp"
#include <vector>
#include <limits>
#include <queue>
#include <vector>
#include <Eigen/SparseCore>
#include <algorithm>

namespace lmb {
    using namespace lap;
    static const double LARGE = std::numeric_limits<double>::max();
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CostMatrix;

    void allbut(const Eigen::MatrixXd& from, Eigen::MatrixXd& to, const unsigned row, const unsigned col)
    {
        unsigned rows = from.rows();
        unsigned cols = from.cols();

        to.resize(rows - 1, cols - 1);

        if (row > 0 && col > 0) {
            to.block(0, 0, row, col) = from.block(0, 0, row, col);
        }
        if (row > 0 && col < cols - 1) {
            to.block(0, col, row, cols - col - 1) = from.block(0, col + 1, row, cols - col - 1);
        }
        if (row < rows - 1 && col > 0) {
            to.block(row, 0, rows - row - 1, col) = from.block(row + 1, 0, rows - row - 1, col);
        }
        if (row < rows - 1 && col < cols - 1) {
            to.block(row, col, rows - row - 1, cols - col - 1) = from.block(row + 1, col + 1, rows - row - 1, cols - col - 1);
        }
    }

    void allbut(const Eigen::Matrix<double, Eigen::Dynamic, 1>& from, Eigen::Matrix<double, Eigen::Dynamic, 1>& to, const unsigned row) {
        unsigned rows = from.rows();

        to.resize(rows - 1, 1);

        if (row > 0) {
            to.block(0, 0, row, 1) = from.block(0, 0, row, 1);
        }
        if (row < rows - 1) {
            to.block(row, 0, rows - row - 1, 1) = from.block(row + 1, 0, rows - row - 1, 1);
        }
    }

    void allbut(const Eigen::Matrix<double, 1, Eigen::Dynamic>& from, Eigen::Matrix<double, 1, Eigen::Dynamic>& to, const unsigned col) {
        unsigned cols = from.cols();

        to.resize(1, cols - 1);

        if (col > 0) {
            to.block(0, 0, 1, col) = from.block(0, 0, 1, col);
        }
        if (col < cols - 1) {
            to.block(0, col, 1, cols - col - 1) = from.block(0, col + 1, 1, cols - col - 1);
        }
    }

    struct State {
        Eigen::MatrixXd C;
        Slack u, v;
        double cost;
        bool solved;
        Assignment solution;
        std::vector<unsigned> bound;

        State()
        : solved(false)
        {}

        State(const Eigen::MatrixXd& C_)
        : C(C_),
          u(C_.rows()),
          v(C_.cols()),
          cost(0),
          solved(false),
          solution(C_.rows())
        {
            v.setZero();
        }

        State bind(const unsigned i, const unsigned j) {
            State s;
            allbut(s.C, C, i, j);
            allbut(s.u, u, i);
            allbut(s.v, v, j);
            s.solution = solution;
            s.bound = bound;
            s.bound.push_back(i);
            std::sort(s.bound.begin(), s.bound.end());

            s.estimate_cost();
            return s;
        }

        State remove(const unsigned i, const unsigned j) {
            State s;
            C = s.C;
            C(i, j) = LARGE;
            s.bound = bound;
            s.solution = solution;
            s.estimate_cost();
            return s;
        }

        void estimate_cost() {
        }

        void solve() {
            Assignment res(C.rows());
            lap::lap(C, res, u, v, cost);
            unsigned k = 0;
            auto it = bound.begin();
            for (unsigned i = 0; i < res.rows(); ++i) {
                while(it != bound.end() && k == *it) {
                    ++k;
                    ++it;
                }
                solution[k] = res[i];
                ++k;
            }
        }
    };

    template<typename CostMatrix>
    class Murty {
        public:
            Murty(const CostMatrix& C_)
            : C(C_) {
            }

            void draw(Assignment&, double&) {
            }

        private:
            CostMatrix C;
            std::priority_queue<State> queue;
    };
}
