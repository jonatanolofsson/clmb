// Copyright 2018 Jonatan Olofsson
#pragma once
#include <vector>
#include <utility>
#include "omp.hpp"
#include "RTree.h"
#include "target.hpp"

namespace lmb {
template<typename Target, typename R>
bool rtree_callback(Target* t, void* result) {
    reinterpret_cast<R*>(result)->emplace_back(t);
    return true;
}

template<typename PDF>
struct TargetTree_ {
    using Self = TargetTree_<PDF>;
    using Target = lmb::Target_<PDF>;
    using Targets = lmb::Targets_<PDF>;

    Targets targets;
    RTree<Target*, double, 2> tree;
    unsigned id_counter;

    TargetTree_()
    : id_counter(0) {
#ifndef NOPAR
        omp_init_lock(&writelock);
#endif
    }

    ~TargetTree_() {
        for (auto t : targets) {
            delete t;
        }
    }

    TargetTree_(const TargetTree_&) = delete;
    TargetTree_ operator=(const TargetTree_&) = delete;

    void remove(Target* const t) {
        auto llaabbox = t->llaabbox();
        tree.Remove(llaabbox.min, llaabbox.max, t);
    }

    void replace(Target* const t, double rlim = 0.0) {
        if (t->r < rlim) {
            targets.erase(std::remove(targets.begin(),
                                      targets.end(),
                                      t),
                          targets.end());
            delete t;
        } else {
            auto llaabbox = t->llaabbox();
            tree.Insert(llaabbox.min, llaabbox.max, t);
        }
    }

    void new_target(double r, PDF&& p, double time) {
        Target* t = new Target(r, std::forward<PDF>(p));
        targets.push_back(t);
        t->id = id_counter++;
        t->t = time;
        auto llaabbox = t->llaabbox();
        //std::cout << "New target: " << *t << ": " << llaabbox << std::endl;
        tree.Insert(llaabbox.min, llaabbox.max, t);
    }

#ifndef NOPAR
    omp_lock_t writelock;
#endif
    void lock() {
#ifndef NOPAR
        omp_set_lock(&writelock);
#endif
    }
    void unlock() {
#ifndef NOPAR
        omp_unset_lock(&writelock);
#endif
    }

    Targets query(const AABBox& aabbox) const {
        Targets result;
        tree.Search(aabbox.min,
                    aabbox.max,
                    rtree_callback<Target, Targets>,
                    reinterpret_cast<void*>(&result));

        // FIXME: Wrap-around
        return result;
    }
};
}  // namespace lmb
