#include "gm.hpp"
#include "lmb.hpp"
#include "models.hpp"
#include "pf.hpp"
#include "sensors.hpp"
#include <Eigen/Core>
#include <iostream>
#include <vector>

using namespace lmb;

int main() {
    //std::cout << "Press enter to run" << std::endl;
    //std::cin.get();
    typedef SILMB<GM<4>> Tracker;
    //typedef SILMB<PF<4>> Tracker;
    Tracker lmb;
    CV<Tracker::Target> model;
    double kappa = 0.01;
    GaussianReport::Covariance P(2, 2); P = Eigen::Matrix2d::Identity();
    PositionSensor<Tracker::Target> s;
    s.lambdaB = 0.5;
    for (double t = 0.0; t < 12; t += 1.0) {
        //std::cout << std::endl << ":::::::::::::::::::::::::::::::::" << std::endl << t << std::endl << ":::::::::::::::::::::::::::::::::" << std::endl;
        std::vector<GaussianReport> zs({
            GaussianReport(Eigen::Vector2d({t, t}), P, kappa),
            GaussianReport(Eigen::Vector2d({t, 10 - t}), P, kappa)
        });
        lmb.predict<CV<Tracker::Target>>(model, s.fov.aabbox(), t);
        lmb.correct(zs, s, t);
        std::cout << "{\"t\":" << t
                  << ",\"scan\":" << zs
                  << ",\"post\":" << lmb
                  << "}" << std::endl;
    }
}
