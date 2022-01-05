// Copyright 2018 Jonatan Olofsson
#pragma once

#include <Eigen/SparseCore>
#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <tuple>
#include <vector>
#include <limits>
#include "rlapjv.h"

namespace lap {
//using Assignment = Eigen::Matrix<int, Eigen::Dynamic, 1>;
//using Dual = Eigen::Matrix<double, Eigen::Dynamic, 1>;
//static const double inf = std::numeric_limits<double>::infinity();

class EmptyQueue: public std::exception {
 public:
    virtual const char* what() const throw() {
        return "Queue empty.";
    }
};

void allbut(const CostMatrix& from, CostMatrix& to, const unsigned row, const unsigned col) {
    unsigned rows = from.rows();
    unsigned cols = from.cols();

    to.resize(rows - 1, cols - 1);

    if (row > 0 && col > 0) {
        to.block(0, 0, row, col) = from.block(0, 0, row, col);
    }
    if (row > 0 && col < cols - 1) {
        to.block(0, col, row, cols - col - 1) =
            from.block(0, col + 1, row, cols - col - 1);
    }
    if (row < rows - 1 && col > 0) {
        to.block(row, 0, rows - row - 1, col) =
            from.block(row + 1, 0, rows - row - 1, col);
    }
    if (row < rows - 1 && col < cols - 1) {
        to.block(row, col, rows - row - 1, cols - col - 1) =
            from.block(row + 1, col + 1, rows - row - 1, cols - col - 1);
    }
}

void allbut(const Eigen::Matrix<double, Eigen::Dynamic, 1>& from, Eigen::Matrix<double, Eigen::Dynamic, 1>& to, const unsigned row) {
    unsigned rows = from.rows();

    to.resize(rows - 1, 1);

    if (row > 0) {
        to.block(0, 0, row, 1) = from.block(0, 0, row, 1);
    }
    if (row < rows - 1) {
        to.block(row, 0, rows - row - 1, 1) =
            from.block(row + 1, 0, rows - row - 1, 1);
    }
}

void allbut(const Eigen::Matrix<double, 1, Eigen::Dynamic>& from, Eigen::Matrix<double, 1, Eigen::Dynamic>& to, const unsigned col) {
    unsigned cols = from.cols();

    to.resize(1, cols - 1);

    if (col > 0) {
        to.block(0, 0, 1, col) = from.block(0, 0, 1, col);
    }
    if (col < cols - 1) {
        to.block(0, col, 1, cols - col - 1) =
            from.block(0, col + 1, 1, cols - col - 1);
    }
}


struct MurtyState {
    using Duallist = std::vector<std::tuple<double, unsigned, unsigned>>;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> C;
    Dual u, v;
    double cost;
    double boundcost;
    bool solved;
    Assignment solution;
    Assignment res;
    std::vector<unsigned> rmap, cmap;

    explicit MurtyState(const MurtyState* s)
    :
      cost(s->cost),
      boundcost(s->boundcost),
      solved(false),
      solution(s->solution),
      rmap(s->rmap),
      cmap(s->cmap) {}

    explicit MurtyState(const CostMatrix& C_)
    : C(C_),
      u(C_.rows()),
      v(C_.cols()),
      cost(0),
      boundcost(0),
      solved(false),
      solution(C_.rows()) {
        v.setZero();
        rmap.resize(C.rows());
        cmap.resize(C.cols());
        for (unsigned i = 0; i < C.rows(); ++i) { rmap[i] = i; }
        for (unsigned j = 0; j < C.cols(); ++j) { cmap[j] = j; }
    }

    auto partition_with(const unsigned i, const unsigned j) {
        auto s = std::make_shared<MurtyState>(this);

        allbut(C, s->C, i, j);
        allbut(u, s->u, i);
        allbut(v, s->v, j);
        //std::cout << "            Binding (" << i << "," << j << ") (" << C(i, j) << ") [" << s->C.rows() << "x" << s->C.cols() << "]" << std::endl;
        s->rmap.erase(s->rmap.begin() + i);
        s->cmap.erase(s->cmap.begin() + j);
        s->boundcost += C(i, j);

        return s;
    }

    auto partition_without(const unsigned i, const unsigned j, const double dual) {
        auto s = std::make_shared<MurtyState>(this);
        s->C = C;
        s->u = u;
        s->v = v;
        s->remove(i, j, dual);
        //std::cout << "            Removing (" << i << "," << j << ") (" << C(i, j) << ") [" << s->C.rows() << "x" << s->C.cols() << "]" << std::endl;

        return s;
    }

    void remove(const unsigned i, const unsigned j, const double dual) {
        //std::cout << " ::::::::::  Remove ::::::::::" << std::endl;
        //std::cout << "Dual u: " << std::endl << u.replicate(1, C.cols()) << std::endl;
        //std::cout << "Dual v: " << std::endl << v.replicate(1, C.rows()).transpose() << std::endl;
        //std::cout << "Old C: " << std::endl << C << std::endl;
        //std::cout << "Dual: " << std::endl << u.replicate(1, C.cols()) + v.replicate(1, C.rows()).transpose() << std::endl;
        //std::cout << "Reduced C: " << std::endl << C - u.replicate(1, C.cols()) - v.replicate(1, C.rows()).transpose() << std::endl;
        //std::cout << "            Removing (" << i << "," << j << ") (" << C(i, j) << ") [" << C.rows() << "x" << C.cols() << "] raising cost " << cost << " + " << dual << " = ";
        C(i, j) = lap::inf;
        solved = false;
        cost += dual;
        //std::cout << cost << std::endl;
        //std::cout << "New C: " << std::endl << C << std::endl;
    }

    bool solve() {
        //std::cout << "Solving (" << cost << "): " << std::endl << C << std::endl;
        res.resize(C.rows());
        auto lapped = lap(C, res, u, v);
        solved = true;
        //std::cout << "Dual: " << u.transpose() << ", " << v.transpose() << std::endl;
        //std::cout << "u: " << u.transpose() << std::endl;
        //std::cout << "v: " << v.transpose() << std::endl;
        //std::cout << "u.rep: " << std::endl << u.replicate(1, C.cols()) << std::endl;
        //std::cout << "v.rep: " << std::endl << v.replicate(1, C.cols()).transpose() << std::endl;
        //std::cout << "Dual: " << std::endl << u.replicate(1, C.cols()) + v.replicate(1, C.rows()).transpose() << std::endl;
        //std::cout << "Reduced C: " << std::endl << C - u.replicate(1, C.cols()) - v.replicate(1, C.rows()).transpose() << std::endl;
        //std::cout << " >>>  Prior cost: " << cost << " / " << boundcost;
        if (!lapped) {
            //std::cout << " ISINF!!!!!!!!!!!" << std::endl;
            return false;
            //std::exit(1);
        }
        //std::cout << " NOINF!!!!!!!!!!!" << std::endl;
        cost = boundcost;
        for (unsigned i = 0; i < res.rows(); ++i) {
            solution[rmap[i]] = cmap[res[i]];
            cost += C(i, res[i]);
            //std::cout << "Dualed: " << i << ":" << res[i] << ": " << C(i, res[i]) - u[i] - v[res[i]] << std::endl;
        }
        //std::cout << " >>>  Posterior cost: " << cost << std::endl;
        //std::cout << "Solution: [" << res.transpose() << "] " << cost << std::endl;
        //auto r = (C - u.replicate(1, C.cols()) - v.replicate(1, C.rows()).transpose()).array();
        //if ((r < 0).any()) {
            //std::cout << "Reduced cost < 0" << std::endl;
            ////std::exit(1);
        //}
        return (cost < lap::inf);
    }

    MurtyState::Duallist sort_by_dual() const {
        //std::cout << " ::::::::::  sort_by_dual ::::::::::" << std::endl;
        //std::cout << "C: " << std::endl << C << std::endl;
        //std::cout << "u: " << std::endl << u.transpose() << std::endl;
        //std::cout << "v: " << std::endl << v.transpose() << std::endl;
        //std::cout << "Reduced C: " << std::endl << C - u.replicate(1, C.cols()) - v.replicate(1, C.rows()).transpose() << std::endl;
        //std::cout << "res: " << res.transpose() << std::endl;
        //std::cout << "Cost: " << cost << std::endl;
        MurtyState::Duallist mdual(C.rows());
        double h;
        for (unsigned i = 0; i < C.rows(); ++i) {
            mdual[i] = {lap::inf, i, res[i]};
            for (unsigned j = 0; j < C.cols(); ++j) {
                if (static_cast<int>(j) == res[i]) {
                    continue;
                }
                if (C(i, j) >= lap::inf) {
                    continue;
                }

                h = C(i, j) - u[i] - v[j];
                if (h < std::get<0>(mdual[i])) {
                    //std::cout << "Found better dual: " << i << "," << j << ": " << C(i, j) << " - " << u[i] << " - " << v[j] << " = " << h << std::endl;
                    std::get<0>(mdual[i]) = h;
                }
            }
        }
        std::sort(mdual.rbegin(), mdual.rend());
        //std::cout << "Dual: " << std::endl;
        //for (unsigned i = 0; i < C.rows(); ++i) {
            //std::cout << "    (" << std::get<0>(mdual[i]) << ", " << std::get<1>(mdual[i]) << ", " << std::get<2>(mdual[i])  << ")" << std::endl;
        //}
        return mdual;
    }
};

using MurtyStatePtr = std::shared_ptr<MurtyState>;
struct MurtyStatePtrCompare {
    bool operator()(const MurtyStatePtr& a, const MurtyStatePtr& b) {
        if (a->cost > b->cost) {
            return true;
        } else if (a->cost == b->cost) {
            return a->C.rows() > b->C.rows();
        } else {
            return false;
        }
    }
};

class Murty {
 private:
    double offset;

 public:
    explicit Murty(CostMatrix C) {
        double min = C.minCoeff();
        offset = 0;
        if (min < 0) {
            C.array() -= min;
            offset = min * C.rows();
        }
        queue.emplace(std::make_shared<MurtyState>(C));
    }

    void get_partition_index(const MurtyState::Duallist& partition_order, MurtyState::Duallist::iterator p, unsigned& i, unsigned& j) {
        i = std::get<1>(*p);
        j = std::get<2>(*p);
        for (auto pp = partition_order.begin(); pp != p; ++pp) {
            if (std::get<1>(*pp) < std::get<1>(*p)) { --i; }
            if (std::get<2>(*pp) < std::get<2>(*p)) { --j; }
        }
    }

    using ReturnTuple = std::tuple<bool, double, Assignment>;
    ReturnTuple draw_tuple() {
        Assignment sol;
        double cost;
        bool ok = draw(sol, cost);
        return {ok, cost, sol};
    }

    bool draw(Assignment& sol, double& cost) {
        std::shared_ptr<MurtyState> s;
        unsigned i, j;
        //std::cout << "Draw! Queue size: " << queue.size() << std::endl;

        if (queue.empty()) {
            //std::cout << "Draw! Queue empty!: " << queue.size() << std::endl;
            return false;
        }

        for (s = queue.top(); !s->solved; s = queue.top()) {
            queue.pop();
            if (s->solve()) {
                queue.push(s);
            }
            //else std::cout << "Solve failed" << std::endl;
            if (queue.empty()) {
                //std::cout << "Queue empty 1!" << std::endl;
                return false;
            }
        }
        queue.pop();
        sol = s->solution;
        cost = s->cost + offset;
        //std::cout << "Solution: [" << sol.transpose() << "] " << cost << std::endl;
        //std::cout << "Solution: " << sol.transpose() << std::endl;
        //std::cout << "Cost: " << cost << std::endl;
        //std::cout << "res: " << s->res.transpose() << std::endl;
        //std::cout << "rmap: ";
        //for (auto& r : s->rmap) {
            //std::cout << r << ", ";
        //}
        //std::cout << std::endl;
        //std::cout << "cmap: ";
        //for (auto& c : s->cmap) {
            //std::cout << c << ", ";
        //}
        //std::cout << std::endl;
        //std::cout << s->C << std::endl;
        //std::cout << "Size: " << s->C.rows() << std::endl;

        auto partition_order = s->sort_by_dual();
        auto p = partition_order.begin();
        auto node = s;

        for (; p != partition_order.end(); ++p) {
            get_partition_index(partition_order, p, i, j);
            auto dual = std::get<0>(*p);
            //std::cout << "dual: " << dual << " i: " << i << " j: " << j << std::endl;
            if (dual < lap::inf) {
                queue.push(node->partition_without(i, j, dual));
            }
            node = node->partition_with(i, j);
        }
        return true;
    }

 private:
    std::priority_queue<MurtyStatePtr, std::vector<MurtyStatePtr>, MurtyStatePtrCompare> queue;
};
}
