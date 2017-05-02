#pragma once
#include <vector>
#include "RTree.h"
#include "target.hpp"

namespace lmb {
    template<typename Target, typename R>
    bool rtree_callback(Target* t, void* result) {
        ((R*)result)->emplace_back(t);
        return true;
    }

    template<typename PDF>
    struct TargetTree {
        typedef TargetTree<PDF> Self;
        typedef lmb::Target<PDF> Target;
        typedef lmb::Targets<PDF> Targets;

        Targets targets;
        RTree<Target*, double, 2> tree;
        unsigned id_counter;

        TargetTree()
        : id_counter(0)
        {}

        ~TargetTree() {
            for (auto t : targets) {
                delete t;
            }
        }

        TargetTree(const TargetTree&) = delete;
        TargetTree operator=(const TargetTree&) = delete;

        void remove(Target* const t) {
            //std::cout << "Remove: " << t->id << " : " << t->pdf.aabbox << std::endl;
            tree.Remove(t->pdf.aabbox.min, t->pdf.aabbox.max, t);
        }

        void replace(Target* const t, double rlim = 0.0) {
            //std::cout << "Replace: " << t->id << " : " << t->pdf.aabbox << std::endl;
            if (t->r <= rlim) {
                targets.erase(std::remove(targets.begin(), targets.end(), t), targets.end());
                delete t;
            } else {
                tree.Insert(t->pdf.aabbox.min, t->pdf.aabbox.max, t);
            }
        }

        void new_target(double r, PDF&& p, double time) {
            Target* t = new Target(r, std::forward<PDF>(p));
            targets.push_back(t);
            t->id = id_counter++;
            t->t = time;
            tree.Insert(t->pdf.aabbox.min, t->pdf.aabbox.max, t);
        }

        // FIXME
        void lock() {}
        void unlock() {}

        void query(const AABBox& aabbox, Targets& result) {
            result.clear();
            tree.Search(aabbox.min, aabbox.max, rtree_callback<Target, Targets>, (void*)&result);
        }
    };
}
