// Copyright 2018 Jonatan Olofsson
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include "gm.hpp"
#include "lmb.hpp"
#include "models.hpp"
#include "pf.hpp"
#include "sensors.hpp"

using namespace lmb;
using GaussianReport = GaussianReport_<2>;

int main() {
    //std::cout << "Press enter to run" << std::endl;
    //std::cin.get();
    typedef SILMB<GM<4>> Tracker;
    //typedef SILMB<PF<4>> Tracker;
    Tracker lmb;
    CV<Tracker::Target> model;
    double kappa = 0.01;
    GaussianReport::Covariance P; P = Eigen::Matrix2d::Identity();
    PositionSensor<Tracker::Target> s;
    cf::LL origin; origin << 58.3887657, 15.6965082;
    s.fov.from_gaussian(origin, 20*Eigen::Matrix2d::Identity());
    s.lambdaB = 0.5;
    double last_time = 0;
    for (double t = 0.0; t < 12; t += 1.0) {
        //std::cout << std::endl << ":::::::::::::::::::::::::::::::::" << std::endl << t << std::endl << ":::::::::::::::::::::::::::::::::" << std::endl;
        std::vector<GaussianReport> zs({
            GaussianReport(Eigen::Vector2d({t, t}), P, kappa),
            GaussianReport(Eigen::Vector2d({t, 10 - t}), P, kappa)
        });
        for (auto& r : zs) { r.transform_to_global(origin); }
        lmb.predict<CV<Tracker::Target>>(model, t, last_time);
        last_time = t;
        lmb.correct(s, zs, t);
        for (auto& t : lmb.targettree.targets) { t->transform_to_local(origin); }
        std::cout << "{\"t\":" << t
                  << ",\"scan\":" << zs
                  << ",\"post\":" << lmb
                  << "}" << std::endl;
        for (auto& t : lmb.targettree.targets) { t->transform_to_global(); }
    }
}
