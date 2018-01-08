#pragma once
#include <vector>
#include "RTree.h"
#include "target.hpp"
//#include <omp.h>

namespace lmb {
    template<typename Target, typename R>
    bool rtree_callback(Target* t, void* result) {
        ((R*)result)->emplace_back(t);
        return true;
    }

    template<typename PDF>
    struct TargetTree {
        using Self = TargetTree<PDF>;
        using Target = lmb::Target_<PDF>;
        using Targets = lmb::Targets_<PDF>;

        Targets targets;
        RTree<Target*, double, 2> tree;
        unsigned id_counter;

        TargetTree()
        : id_counter(0)
        {
#ifdef _OPENMP
            omp_init_lock(&writelock);
#endif
        }

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

#ifdef _OPENMP
        omp_lock_t writelock;
#endif
        void lock() {
#ifdef _OPENMP
            omp_set_lock(&writelock);
#endif
        }
        void unlock() {
#ifdef _OPENMP
            omp_unset_lock(&writelock);
#endif
        }

        Targets query(const AABBox& aabbox) const {
            Targets result;
            tree.Search(aabbox.min, aabbox.max, rtree_callback<Target, Targets>, (void*)&result);
            return result;
        }
    };
}
