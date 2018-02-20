// Copyright 2018 Jonatan Olofsson
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "cf.hpp"
namespace py = pybind11;

PYBIND11_MODULE(cf, m) {
    m.def("nedrot", cf::nedrot<cf::LLA>);
    m.def("lla2ecef", cf::lla2ecef<cf::LLA>);
    m.def("ecef2ned", cf::ecef2ned<cf::ECEF, cf::LLA>);
    m.def("lla2ned", cf::lla2ned<cf::LLA, cf::LLA>);
    m.def("ecef2lla", cf::ecef2lla<cf::ECEF>);
    m.def("ned2ecef", cf::ned2ecef<cf::NED, cf::LLA>);
    m.def("ned2lla", cf::ned2lla<cf::NED, cf::LLA>);
}
