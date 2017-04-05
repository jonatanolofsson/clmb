#pragma once

#include "lap.hpp"
#include <vector>
#include <limits>
#include <queue>
#include <Eigen/SparseCore>
#include <algorithm>

namespace lmb {
    using namespace lap;
    static const double LARGE = std::numeric_limits<double>::max();
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CostMatrix;

    void remove(Eigen::MatrixXd& matrix, unsigned row, unsigned col)
    {
        unsigned rows = matrix.rows();
        unsigned cols = matrix.cols();

        if(row < rows - 1) {
            matrix.block(row, 0, rows - row - 1, cols) = matrix.block(row + 1, 0, rows - row - 1, cols);
        }
        if(col < cols - 1) {
            matrix.block(0, col, rows - 1, cols - col - 1) = matrix.block(0, col + 1, rows - 1, cols - col - 1);
        }

        matrix.resize(rows - 1, cols - 1);
    }


    struct State {
        const CostMatrix& C;
        Eigen::SparseMatrix<bool> mask;
        std::vector<int> rows;
        std::vector<int> cols;

        State(const CostMatrix& C_)
        : C(C_), rows(C_.rows()), cols(C_.cols()) {
            mask.resize(C.rows(), C.cols());
            for (int i = 0; i < (int)C.rows(); ++i) rows[i] = i;
            for (int j = 0; j < (int)C.cols(); ++j) cols[j] = j;
        }

        State(const State& s) : C(s.C) {
            mask = s.mask;
            rows = s.rows;
            cols = s.cols;
        }

        State remove(const int row, const int col) {
            State s(*this);
            s.mask.insert(row, col) = true;
            return s;
        }

        State bind(const int row, const int col) {
            State s(*this);
            auto pos = std::find(s.rows.begin(), s.rows.end(), row);
            if (pos != s.rows.end()) s.rows.erase(pos);
            pos = std::find(s.cols.begin(), s.cols.end(), col);
            if (pos != s.cols.end()) s.cols.erase(pos);
            return s;
        }
    };

    template<typename T>
    struct MaskedRowVector {
        const State& s;
        T u;

        MaskedRowVector(const State& s_) : s(s_), u(s.C.rows()) {}

        typename T::Scalar& operator[](const int i) {
            return u[s.rows[i]];
        }

        int rows() const {
            return s.C.rows();
        }

        void setZero() { u.setZero(); }
    };

    template<typename T>
    struct MaskedColVector {
        const State& s;
        T v;

        MaskedColVector(const State& s_) : s(s_), v(s.C.cols()) {
            v.setZero();
        }

        typename T::Scalar& operator[](const int j) {
            return v[s.cols[j]];
        }

        int cols() const {
            return s.C.cols();
        }

        void setZero() { v.setZero(); }
    };

    struct MaskedMatrix {
        const State& s;

        MaskedMatrix(const State& s_) : s(s_) {}

        double operator()(const int i, const int j) const {
            return s.mask.coeff(i, j) ? LARGE : s.C(s.rows[i], s.cols[j]);
        }

        int rows() const {
            return s.C.rows();
        }

        int cols() const {
            return s.C.cols();
        }
    };

    template<typename CostMatrix>
    class Murty {
        public:
            Murty(const CostMatrix& C_)
            : C(C_) {
                State initial_state(C);
                initial_state.rows.resize(C.rows());
                for (int i = 0; i < (int)initial_state.rows.size(); ++i) {
                    initial_state.rows[i] = i;
                }
                initial_state.cols.resize(C.cols());
                for (int j = 0; j < (int)initial_state.cols.size(); ++j) {
                    initial_state.cols[j] = j;
                }
            }

            void draw(Assignment&, double&) {
            }

        private:
            CostMatrix C;
            std::priority_queue<State> queue;
    };
}
