#include <Eigen/Core>
#include "lmb.hpp"
#include "gm.hpp"
#include "sensors.hpp"
#include "models.hpp"
#include <iostream>

using namespace lmb;

int main() {
    //std::cout << "Press enter to run" << std::endl;
    //std::cin.get();
    typedef SILMB<GM<4>> Filter;
    Filter lmb;
    lmb.lambdaB = 0.5;
    double kappa = 0.0001;
    Report::Covariance P(2, 2); P = Eigen::Matrix2d::Identity();
    PositionSensor<Filter::Target> s;
    for (double t = 0.0; t < 12; t += 1.0) {
        //std::cout << std::endl << ":::::::::::::::::::::::::::::::::" << std::endl << t << std::endl << ":::::::::::::::::::::::::::::::::" << std::endl;
        std::vector<Report> zs({
            Report(Eigen::Vector2d({t, t}), P, kappa),
            Report(Eigen::Vector2d({t, 10 - t}), P, kappa)
        });
        lmb.predict<CV<3000>>(s.aabbox, t);
        lmb.correct(zs, s, t);
        std::cout << "{\"t\":" << t << ",\"scan\":" << zs << ",\"post\":" << lmb << "}" << std::endl;
    }
}
