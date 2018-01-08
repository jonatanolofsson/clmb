#include <Eigen/Core>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <queue>
#include <signal.h>
#include <string>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lmb.hpp"
#include "gaussian.hpp"
#include "gm.hpp"
#include "models.hpp"
#include "params.hpp"
#include "pf.hpp"
#include "sensors.hpp"

namespace py = pybind11;
using namespace lmb;

struct HaltException : public std::exception {};

static const int STATES = 4;
using Tracker = SILMB<GM<STATES>>;
using Gaussian = Gaussian_<STATES>;
using TargetSummary = TargetSummary_<STATES>;


PYBIND11_MODULE(lmb, m) {
    using PosSensor = PositionSensor<Tracker::Target>;
    py::class_<PosSensor>(m, "PositionSensor")
        .def(py::init())
        .def_readwrite("lambdaB", &PosSensor::lambdaB)
        .def_readwrite("pD", &PosSensor::pD_);
    py::class_<NM<Tracker::Target>>(m, "NM")
        .def(py::init())
        .def(py::init<double, Eigen::Matrix4d>())
        .def_readwrite("pS", &NM<Tracker::Target>::pS)
        .def_readwrite("Q", &NM<Tracker::Target>::Q);
    py::class_<CV<Tracker::Target>>(m, "CV")
        .def(py::init())
        .def(py::init<double, double>())
        .def_readwrite("pS", &CV<Tracker::Target>::pS)
        .def_readwrite("q", &CV<Tracker::Target>::q);
    py::class_<BBox>(m, "BBox")
        .def(py::init())
        .def_readwrite("corners", &BBox::corners)
        .def("__repr__", &print<BBox>);
    py::class_<Tracker::Gaussian>(m, "Gaussian")
        .def(py::init<Tracker::Gaussian::State, Tracker::Gaussian::Covariance, double>())
        .def_readwrite("w", &Tracker::Gaussian::w)
        .def_readwrite("m", &Tracker::Gaussian::m)
        .def_readwrite("P", &Tracker::Gaussian::P)
        .def("__repr__", &print<Tracker::Gaussian>);
    py::class_<TargetSummary>(m, "Target")
        .def_readonly("m", &TargetSummary::m)
        .def_readonly("P", &TargetSummary::P)
        .def_readonly("r", &TargetSummary::w)
        .def_readonly("id", &TargetSummary::id)
        .def("__repr__", &print<TargetSummary>);
    py::class_<GaussianReport>(m, "GaussianReport")
        .def(py::init<GaussianReport::Measurement, GaussianReport::Covariance, double>())
        .def_readwrite("z", &GaussianReport::z)
        .def_readwrite("R", &GaussianReport::R)
        .def_readwrite("kappa", &GaussianReport::kappa)
        .def("__repr__", &print<GaussianReport>);
    py::class_<Params>(m, "Params")
        .def_readwrite("w_lim", &Params::w_lim)
        .def_readwrite("r_lim", &Params::r_lim)
        .def_readwrite("nhyp_max", &Params::nhyp_max)
        .def_readwrite("rB_max", &Params::rB_max)
        .def_readwrite("nstd", &Params::nstd);
    py::class_<Tracker>(m, "Tracker")
        .def(py::init())
        .def_readwrite("params", &Tracker::params)
        .def("correct", &Tracker::template correct<PosSensor>, py::call_guard<py::gil_scoped_release>())
        .def("predict", (void (Tracker::*)(NM<Tracker::Target>&, double)) &Tracker::predict<NM<Tracker::Target>>, py::call_guard<py::gil_scoped_release>())
        .def("predict", (void (Tracker::*)(CV<Tracker::Target>&, double)) &Tracker::predict<CV<Tracker::Target>>, py::call_guard<py::gil_scoped_release>())
        .def("enof_targets", &Tracker::enof_targets)
        .def("nof_targets", &Tracker::nof_targets)
        .def("get_targets", &Tracker::get_targets)
        .def("ospa", &Tracker::ospa)
        .def("gospa", &Tracker::gospa);
}
