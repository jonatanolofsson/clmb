#pragma once

#include "lap.hpp"
#include <vector>
#include <queue>
#include <set>
#include <Eigen/SparseCore>
#include <algorithm>
#include <tuple>
#include <exception>
#include <iostream>

namespace lmb {
    using namespace lap;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CostMatrix;

    class EmptyQueue: public std::exception {
    public:
        virtual const char* what() const throw() {
            return "Queue empty.";
        }
    };

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


    struct MurtyState {
        typedef std::vector<std::tuple<double, unsigned, unsigned>> Slacklist;
        Eigen::MatrixXd C;
        Slack u, v;
        double cost;
        double boundcost;
        bool solved;
        Assignment solution;
        Assignment res;
        std::vector<unsigned> rmap, cmap;

        MurtyState(const MurtyState* s)
        :
          cost(s->cost),
          boundcost(s->boundcost),
          solved(false),
          solution(s->solution),
          rmap(s->rmap),
          cmap(s->cmap)
        {}

        MurtyState(const Eigen::MatrixXd& C_)
        : C(C_),
          u(C_.rows()),
          v(C_.cols()),
          cost(0),
          boundcost(0),
          solved(false),
          solution(C_.rows())
        {
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
            //std::cout << "            Binding (" << rmap[i] << "," << cmap[j] << ") (" << C(i, j) << ") [" << s->C.rows() << "x" << s->C.cols() << "]" << std::endl;
            s->rmap.erase(s->rmap.begin() + i);
            s->cmap.erase(s->cmap.begin() + j);
            s->boundcost += C(i, j);

            return s;
        }

        auto partition_without(const unsigned i, const unsigned j, const double slack) {
            auto s = std::make_shared<MurtyState>(this);
            s->C = C;
            s->u = u;
            s->v = v;
            s->remove(i, j, slack);

            return s;
        }

        void remove(const unsigned i, const unsigned j, const double slack) {
            //std::cout << "            Removing (" << rmap[i] << "," << cmap[j] << ") (" << C(i, j) << ") [" << C.rows() << "x" << C.cols() << "] rasing cost " << cost << " -> ";
            C(i, j) = inf;
            solved = false;
            cost += slack;
            //std::cout << cost << std::endl;
            //std::cout << C << std::endl;
        }

        bool solve() {
            //std::cout << "Solving (" << cost << "): " << std::endl << C << std::endl;
            res.resize(C.rows());
            lap::lap(C, res, u, v);
            //std::cout << "rmap: ";
            //for (auto& r : rmap) {
                //std::cout << r << ", ";
            //}
            //std::cout << std::endl;
            //std::cout << "cmap: ";
            //for (auto& c : cmap) {
                //std::cout << c << ", ";
            //}
            //std::cout << std::endl;
            cost = boundcost;
            for (unsigned i = 0; i < res.rows(); ++i) {
                solution[rmap[i]] = cmap[res[i]];
                cost += C(i, res[i]);
            }
            //std::cout << "Solution: [" << res.transpose() << "] " << cost << std::endl;
            solved = true;
            return (cost < inf);
        }

        MurtyState::Slacklist minslack() const {
            std::vector<std::tuple<double, unsigned, unsigned>> mslack(C.rows());
            double h;
            for (unsigned i = 0; i < C.rows(); ++i) {
                mslack[i] = {C(i, 0) - u[i] - v[0], i, res[i]};
                for (unsigned j = 1; j < C.cols(); ++j) {
                    if ((int)j == res[i]) {
                        continue;
                    }

                    h = C(i, j) - u[i] - v[j];
                    if (h < std::get<0>(mslack[i])) {
                        mslack[i] = {h, i, res[i]};
                    }
                }
            }
            std::sort(mslack.rbegin(), mslack.rend());
            return mslack;
        }
    };

    typedef std::shared_ptr<MurtyState> MurtyStatePtr;
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
        public:
            double offset;

            Murty(CostMatrix C) {
                double min = C.minCoeff();
                C.array() -= min;
                offset = min * C.rows();
                queue.emplace(std::make_shared<MurtyState>(C));
            }

            void get_partition_index(const MurtyState::Slacklist& partition_order, MurtyState::Slacklist::iterator p, unsigned& i, unsigned& j) {
                i = std::get<1>(*p);
                j = std::get<2>(*p);
                for (auto pp = partition_order.begin(); pp != p; ++pp) {
                    if (std::get<1>(*pp) < std::get<1>(*p)) { --i; }
                    if (std::get<2>(*pp) < std::get<2>(*p)) { --j; }
                }
            }

            bool draw(Assignment& sol, double& cost) {
                std::shared_ptr<MurtyState> s;
                //std::cout << "Queue size: " << queue.size() << std::endl;

                if (queue.empty()) {
                    return false;
                }

                for (s = queue.top(); !s->solved; s = queue.top()) {
                    queue.pop();
                    if(s->solve()) {
                        queue.push(s);
                    }
                    if (queue.empty()) {
                        return false;
                    }
                }
                queue.pop();
                sol = s->solution;
                cost = s->cost + offset;
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

                auto partition_order = s->minslack();

                if (partition_order.size() == 1) {
                    auto p = partition_order.begin();
                    if (s->cost + std::get<0>(*p) < inf) {
                        unsigned i = std::get<1>(*p);
                        unsigned j = std::get<2>(*p);
                        s->remove(i, j, std::get<0>(*p));
                        queue.push(s);
                    }
                    return (cost < inf);
                }

                partition_order.pop_back();
                auto p = partition_order.begin();
                unsigned i = std::get<1>(*p);
                unsigned j = std::get<2>(*p);
                auto node = s->partition_with(i, j);
                if (s->cost + std::get<0>(*p) < inf) {
                    s->remove(i, j, std::get<0>(*p));
                    queue.push(s);
                }
                ++p;

                if (p == partition_order.end()) {
                    return (cost < inf);
                }

                for (; p + 1 != partition_order.end(); ++p) {
                    get_partition_index(partition_order, p, i, j);
                    if (node->cost + std::get<0>(*p) < inf) {
                        queue.push(node->partition_without(i, j, std::get<0>(*p)));
                    }
                    node = node->partition_with(i, j);
                }
                if (node->cost + std::get<0>(*p) < inf) {
                    get_partition_index(partition_order, p, i, j);
                    node->remove(i, j, std::get<0>(*p));
                    queue.push(node);
                }
                return (cost < inf);
            }

        private:
            std::priority_queue<MurtyStatePtr, std::vector<MurtyStatePtr>, MurtyStatePtrCompare> queue;
    };
}
