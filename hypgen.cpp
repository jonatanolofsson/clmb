#include <vector>
#include "hypgen.hpp"
#include "graph/graph.h"
#include "graph/linear_assignment.h"

namespace lmb {
    using namespace ::operations_research;
    typedef StaticGraph<> Graph;

    Murty::Murty(const CostMatrix C) {
        size = C.cols();
        state.C = C;
    }

    void Murty::draw(Assignment&, double&) {
    }

    void lap(const CostMatrix& cost, Assignment& res, double& total_cost) {
        auto R = cost.rows();
        auto C = cost.cols();

        unsigned num_arcs = R * C;

        std::vector<double> arc_costs(num_arcs);

        Graph graph(R + C, num_arcs);
        for(unsigned r = 0; r < R; ++r) {
            for(unsigned c = 0; c < C; ++c) {
                graph.AddArc(r, R + c);
                arc_costs[C * r + c] = cost(r, c);
            }
        }

        {
            std::vector<typename Graph::ArcIndex> arc_permutation;
            graph.Build(&arc_permutation);
            Permute(arc_permutation, &arc_costs);
        }

        LinearSumAssignment<Graph> a(graph, R);
        for(unsigned r = 0; r < R; ++r) {
            for(unsigned c = 0; c < C; ++c) {
                a.SetArcCost(C * r + c, arc_costs[C * r + c]);
            }
        }

        //bool success = a.ComputeAssignment();
        a.ComputeAssignment();

        total_cost = 0;
        for(int r = 0; r < R; ++r) {
            res[r] = a.GetMate(r) - R;
            total_cost += a.GetAssignmentCost(r);

            //std::cout << r << ": [";
            //for(int c = 0; c < C; ++c) {
                //std::cout << a.PartialReducedCost(C * r + c) << ", ";
            //}
            //std::cout << "]" << std::endl;
        }
    }
}
