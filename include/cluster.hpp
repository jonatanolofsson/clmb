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
        //std::cout << "Transforming cluster " << id << " to local" << std::endl;
        origin = center_of_mass();
        //std::cout << "  Origin: " << origin.format(eigenformat) << std::endl;
        //std::cout << "Reports: " << std::endl;
        for (auto& r : reports) {
            //std::cout << "\t" << *r;
            r->transform_to_local(origin);
            //std::cout << " -> " << *r << std::endl;
        }
        //std::cout << "Targets: " << std::endl;
        for (auto& t : targets) {
            //std::cout << "\t" << *t;
            t->transform_to_local(origin);
            //std::cout << " -> " << *t << std::endl;
        }
    }

    void transform_to_global() {
        //std::cout << "Transforming cluster " << id << " to global" << std::endl;
        //std::cout << "  Origin: " << origin.format(eigenformat) << std::endl;
        //std::cout << "Reports: " << std::endl;
        for (auto& r : reports) {
            //std::cout << "\t" << *r;
            r->transform_to_global(origin);
            //std::cout << " -> " << *r << std::endl;
        }
        //std::cout << "Targets: " << std::endl;
        for (auto& t : targets) {
            //std::cout << "\t" << *t;
            t->transform_to_global();
            //std::cout << " -> " << *t << std::endl;
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
