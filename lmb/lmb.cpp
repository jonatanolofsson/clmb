// Copyright 2018 Jonatan Olofsson
#include <signal.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <queue>
#include <string>
#include "lmb.hpp"
#include "gaussian.hpp"
#include "gm.hpp"
#include "pf.hpp"
#include "models.hpp"
#include "params.hpp"
#include "sensors.hpp"

namespace py = pybind11;
using namespace lmb;
using namespace py::literals;

struct HaltException : public std::exception {};

static const int STATES = 4;
using Tracker = SILMB<GM<STATES>>;
//using Tracker = SILMB<PF<STATES>>;
using Gaussian = Gaussian_<STATES>;
using TargetSummary = TargetSummary_<STATES>;
using GaussianReport = GaussianReport_<2>;


PYBIND11_MODULE(lmb, m) {
    using PosSensor = PositionSensor<Tracker::Target>;
    py::class_<PosSensor>(m, "PositionSensor")
        .def(py::init())
        .def_readwrite("fov", &PosSensor::fov)
        .def_readwrite("lambdaB", &PosSensor::lambdaB)
        .def_readwrite("pD", &PosSensor::pD_)
        .def_readwrite("pv", &PosSensor::pv)
        .def_readwrite("kappa", &PosSensor::kappa_);
    py::class_<NM<Tracker::Target>>(m, "NM")
        .def(py::init())
        .def(py::init<double, Eigen::Matrix4d>())
        .def_readwrite("pS", &NM<Tracker::Target>::pS)
        .def_readwrite("Q", &NM<Tracker::Target>::Q);
    py::class_<CV<Tracker::Target>>(m, "CV")
        .def(py::init<double, double>(), "pS"_a=1, "q"_a=1)
        .def_readwrite("pS", &CV<Tracker::Target>::pS)
        .def_readwrite("q", &CV<Tracker::Target>::q);
    py::class_<BBox>(m, "BBox")
        .def(py::init())
        .def(py::init<BBox::Corners>())
        .def_readwrite("corners", &BBox::corners)
        .def("from_gaussian",
             &BBox::template from_gaussian<Eigen::Vector2d, Eigen::Matrix2d>,
             py::arg("mean"), py::arg("cov"), py::arg("nstd") = 2.0)
        .def("nebbox", &BBox::nebbox)
        .def("aabbox", &BBox::aabbox)
        .def("__repr__", &print<BBox>)
        .def(py::pickle(
            [](const BBox& bbox) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(bbox.corners(0), bbox.corners(1), bbox.corners(2), bbox.corners(3),
                                      bbox.corners(4), bbox.corners(5), bbox.corners(6), bbox.corners(7));
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 8) {
                    throw std::runtime_error("Invalid state!");
                }

                /* Create a new C++ instance */
                return BBox((BBox::Corners() <<
                                t[0].cast<double>(), t[2].cast<double>(), t[4].cast<double>(), t[6].cast<double>(),
                                t[1].cast<double>(), t[3].cast<double>(), t[5].cast<double>(), t[7].cast<double>()).finished());
            }
        ));
    py::class_<AABBox>(m, "AABBox")
        .def(py::init())
        .def(py::init<double, double, double, double>())
        //.def("from_gaussian", &BBox::template from_gaussian<Eigen::Vector2d, Eigen::Matrix2d>)  // NOLINT
        .def("neaabbox", &AABBox::neaabbox)
        .def("__repr__", &print<BBox>)
        .def(py::pickle(
            [](const AABBox& aabbox) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(aabbox.min[0], aabbox.min[1],
                                      aabbox.max[0], aabbox.max[1]);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4) {
                    throw std::runtime_error("Invalid state!");
                }

                /* Create a new C++ instance */
                return AABBox(t[0].cast<double>(), t[1].cast<double>(),
                              t[2].cast<double>(), t[3].cast<double>());
            }
        ));
    py::class_<Tracker::Gaussian>(m, "Gaussian")
        .def(py::init<Tracker::Gaussian::State,
                      Tracker::Gaussian::Covariance,
                      double>())
        .def_readwrite("w", &Tracker::Gaussian::w)
        .def_readwrite("m", &Tracker::Gaussian::x)
        .def_readwrite("P", &Tracker::Gaussian::P)
        .def("__repr__", &print<Tracker::Gaussian>);
    py::class_<TargetSummary>(m, "Target")
        .def_readonly("x", &TargetSummary::x)
        .def_readonly("P", &TargetSummary::P)
        .def_readonly("r", &TargetSummary::w)
        .def_readonly("id", &TargetSummary::id)
        .def_readonly("cid", &TargetSummary::cid)
        .def("nebbox", &TargetSummary::nebbox, "origin"_a, "nstd"_a = 2.0)
        .def_readonly("cid", &TargetSummary::cid)
        .def("__repr__", &print<TargetSummary>)
        .def(py::pickle(
            [](const TargetSummary& t) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(t.x, t.P, t.w, t.id, t.cid);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 5) {
                    throw std::runtime_error("Invalid state!");
                }

                /* Create a new C++ instance */
                return TargetSummary(
                    t[0].cast<decltype(TargetSummary::x)>(),
                    t[1].cast<decltype(TargetSummary::P)>(),
                    t[2].cast<decltype(TargetSummary::w)>(),
                    t[3].cast<decltype(TargetSummary::id)>(),
                    t[4].cast<decltype(TargetSummary::cid)>());
            }
        ));
    py::class_<GaussianReport>(m, "GaussianReport")
        .def(py::init<PosSensor&,
                      GaussianReport::State,
                      GaussianReport::Covariance>(), "sensor"_a, "state"_a, "cov"_a)
        .def(py::init<GaussianReport::State,
                      GaussianReport::Covariance,
                      double>(), "state"_a, "cov"_a, "kappa"_a = 1.0)
        .def_readwrite("x", &GaussianReport::x)
        .def_readwrite("P", &GaussianReport::P)
        .def_readwrite("kappa", &GaussianReport::kappa)
        .def_readonly("cid", &GaussianReport::cid)
        .def("nebbox", &GaussianReport::nebbox, "origin"_a, "nstd"_a = 2.0)
        .def("__repr__", &print<GaussianReport>)
        .def(py::pickle(
            [](const GaussianReport& r) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(r.x, r.P, r.kappa, r.cid);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                GaussianReport r(
                    t[0].cast<decltype(GaussianReport::x)>(),
                    t[1].cast<decltype(GaussianReport::P)>(),
                    t[2].cast<decltype(GaussianReport::kappa)>());
                r.cid = t[3].cast<decltype(GaussianReport::cid)>();
                return r;
            }
        ));
    py::class_<Params>(m, "Params")
        .def(py::init())
        .def_readwrite("w_lim", &Params::w_lim)
        .def_readwrite("r_lim", &Params::r_lim)
        .def_readwrite("nhyp_max", &Params::nhyp_max)
        .def_readwrite("rB_max", &Params::rB_max)
        .def_readwrite("nstd", &Params::nstd)
        .def_readwrite("cw_lim", &Params::cw_lim)
        .def_readwrite("cov_lim", &Params::cov_lim)
        .def_readwrite("tscale", &Params::tscale);
    py::class_<Tracker>(m, "Tracker")
        .def(py::init())
        .def(py::init<typename Tracker::Params>())
        .def_readwrite("params", &Tracker::params)
        .def("correct", [](Tracker& tracker, const PosSensor& sensor, typename PosSensor::Scan& scan, double time) {
                tracker.correct<PosSensor>(sensor, scan, time);
                return scan;
        }, py::call_guard<py::gil_scoped_release>())
        .def("predict", (void (Tracker::*)(NM<Tracker::Target>&, double, double)) &Tracker::predict<NM<Tracker::Target>>, py::call_guard<py::gil_scoped_release>())  // NOLINT
        .def("predict", (void (Tracker::*)(CV<Tracker::Target>&, double, double)) &Tracker::predict<CV<Tracker::Target>>, py::call_guard<py::gil_scoped_release>())  // NOLINT
        .def("enof_targets", (double (Tracker::*)())&Tracker::enof_targets)
        .def("enof_targets", (double (Tracker::*)(const AABBox&))&Tracker::enof_targets)
        .def("nof_targets", (unsigned (Tracker::*)(double))&Tracker::nof_targets, "r_lim"_a = 0.7)
        .def("nof_targets", (unsigned (Tracker::*)(const AABBox&, double))&Tracker::nof_targets, "aabbox"_a, "r_lim"_a = 0.7)
        .def("get_targets", (typename Tracker::TargetSummaries (Tracker::*)())&Tracker::get_targets)
        .def("get_targets", (typename Tracker::TargetSummaries (Tracker::*)(const AABBox&))&Tracker::get_targets)
        .def("pos_phd", &Tracker::pos_phd)
        .def("ospa", (double (Tracker::*)(const typename Tracker::TargetStates&, const double, const double))&Tracker::ospa)
        .def("gospa", (double (Tracker::*)(const typename Tracker::TargetStates&, const double, const double))&Tracker::gospa)
        .def_readonly("cluster_ntargets", &Tracker::cluster_ntargets)
        .def_readonly("cluster_nreports", &Tracker::cluster_nreports)
        .def_readonly("cluster_nhyps", &Tracker::cluster_nreports)
        .def_readonly("nof_clusters", &Tracker::nof_clusters);
}
