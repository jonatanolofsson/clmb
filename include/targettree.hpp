// Copyright 2018 Jonatan Olofsson
#pragma once
#include <vector>
#include <utility>
#include "omp.hpp"
#include "RTree.h"
#include "target.hpp"
#include "cf.hpp"

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
    const Params* params;

    explicit TargetTree_(const Params* params_)
    : id_counter(0), params(params_) {}

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

    void remove_all() {
        tree.RemoveAll();
    }

    void erase(Target* const t) {
        targets.erase(std::remove(targets.begin(), targets.end(), t),
                      targets.end());
        delete t;
    }

    void replace(Target* const t) {
        auto llaabbox = t->llaabbox();
        tree.Insert(llaabbox.min, llaabbox.max, t);
    }

    Target* new_target(double r, PDF&& p, const double t0 = 0) {
        Target* t = new Target(r, std::forward<PDF>(p), params, t0);
        targets.push_back(t);
        t->id = id_counter++;
        auto llaabbox = t->llaabbox();
#ifdef DEBUG_OUTPUT
        std::cout << "New target: " << *t << ": " << llaabbox << std::endl;
#endif
        tree.Insert(llaabbox.min, llaabbox.max, t);
        return t;
    }

    Targets query(const AABBox& llaabbox) const {
        Targets result;
        tree.Search(llaabbox.min,
                    llaabbox.max,
                    rtree_callback<Target, Targets>,
                    reinterpret_cast<void*>(&result));

        // FIXME: Wrap-around
        return result;
    }

    Targets query(const cf::LL& point) const {
        Targets result;
        tree.Search(point.data(), point.data(),
                    rtree_callback<Target, Targets>,
                    reinterpret_cast<void*>(&result));

        // FIXME: Wrap-around
        return result;
    }
};
}  // namespace lmb
