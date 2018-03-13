// Copyright 2018 Jonatan Olofsson
#pragma once
#include <Eigen/Eigenvalues>
#include <cmath>
#include <vector>
#include "cf.hpp"
#include "report.hpp"
#include "target.hpp"

namespace lmb {

template<typename Report, typename Target>
struct Cluster_ {
    using Self = Cluster_;
    using Reports = std::vector<Report*>;
    using Targets = std::vector<Target*>;
    cf::LL origin;

    Reports reports;
    Targets targets;
    unsigned id;

    Cluster_(const unsigned id_, const Reports& reports_, const Targets& targets_)  // NOLINT
    : reports(reports_),
      targets(targets_),
      id(id_)
    {}

    explicit Cluster_(const Targets& targets_)
    : targets(targets_) {}

    cf::LL center_of_mass() {
        cf::LL s; s.setZero();
        for (auto r : reports) { s += r->pos(); }
        for (auto t : targets) { s += t->pos(); }
        auto n = reports.size() + targets.size();
        return s / n;
    }

    void transform_to_local() {
#ifdef DEBUG_OUTPUT
        std::cout << "Transforming cluster " << id << " to local" << std::endl;
#endif
        origin = center_of_mass();
#ifdef DEBUG_OUTPUT
        std::cout << "  Origin: " << origin.format(eigenformat) << std::endl;
        std::cout << "Reports: " << std::endl;
#endif
        for (auto& r : reports) {
#ifdef DEBUG_OUTPUT
            std::cout << "\t" << *r;
#endif
            r->transform_to_local(origin);
#ifdef DEBUG_OUTPUT
            std::cout << " -> " << *r << std::endl;
#endif
        }
#ifdef DEBUG_OUTPUT
        std::cout << "Targets: " << std::endl;
#endif
        for (auto& t : targets) {
#ifdef DEBUG_OUTPUT
            std::cout << "\t" << *t;
#endif
            t->transform_to_local(origin);
#ifdef DEBUG_OUTPUT
            std::cout << " -> " << *t << std::endl;
#endif
        }
    }

    void transform_to_global() {
#ifdef DEBUG_OUTPUT
        std::cout << "Transforming cluster " << id << " to global" << std::endl;
        std::cout << "  Origin: " << origin.format(eigenformat) << std::endl;
        std::cout << "Reports: " << std::endl;
#endif
        for (auto& r : reports) {
#ifdef DEBUG_OUTPUT
            std::cout << "\t" << *r;
#endif
            r->transform_to_global(origin);
#ifdef DEBUG_OUTPUT
            std::cout << " -> " << *r << std::endl;
#endif
        }
#ifdef DEBUG_OUTPUT
        std::cout << "Targets: " << std::endl;
#endif
        for (auto& t : targets) {
#ifdef DEBUG_OUTPUT
            std::cout << "\t" << *t;
#endif
            t->transform_to_global();
#ifdef DEBUG_OUTPUT
            std::cout << " -> " << *t << std::endl;
#endif
        }
    }

    void repr(std::ostream& os) const {
        os << "{\"type\":\"CL\"";
        os << ",\"R\":[";
        bool first = true;
        for (auto& r : reports) {
            if (!first) { os << ","; } else { first = false; }
            os << *r;
        }
        os << "],\"T\":[";
        first = true;
        for (auto& t : targets) {
            if (!first) { os << ","; } else { first = false; }
            os << *t;
        }
        os << "]}";
    }
};

template<typename Report, typename Target>
auto& operator<<(std::ostream& os, const Cluster_<Report, Target>& f) {
    f.repr(os);
    return os;
}

}  // namespace lmb
