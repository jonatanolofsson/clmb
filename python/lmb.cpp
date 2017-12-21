#include <Eigen/Core>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <queue>
#include <signal.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lmb.hpp"
#include "gauss.hpp"
#include "gm.hpp"
#include "models.hpp"
#include "params.hpp"
#include "pf.hpp"
#include "sensors.hpp"

namespace py = pybind11;
using namespace lmb;

struct HaltException : public std::exception {};

static const int STATES = 4;
typedef SILMB<GM<STATES>> Tracker;
typedef GaussianComponent<STATES> Gaussian;

struct alignas(16) TargetSummary {
    typedef TargetSummary Self;
    typedef Eigen::Matrix<double, STATES, 1> State;
    typedef Eigen::Matrix<double, STATES, STATES> Covariance;
    State m;
    Covariance P;
    double r;
    unsigned id;

    TargetSummary(){}

    TargetSummary(const State& m_, const Covariance P_, double r_, unsigned id_)
    : m(m_), P(P_), r(r_), id(id_) {}

    void repr(std::ostream& os) const {
        os << "{\"type\":\"T\""
           << ",\"id\":" << id
           << ",\"r\":" << r
           << ",\"m\":" << m.format(eigenformat)
           << ",\"P\":" << P.format(eigenformat)
           << "}";
    }
};

class PyInterface {
    public:
        typedef std::vector<TargetSummary> Summaries;
        Tracker::Params params;

    private:
        double t = 0;
        Tracker lmb;

    public:

        PyInterface()
        : lmb(&params) {}

        template<typename Sensor>
        void correct(Sensor& sensor, std::vector<GaussianReport>& scan) {
            lmb.correct(scan, sensor, t);
        }

        template<typename Model>
        void predict(Model& model, const double t) {
            lmb.predict<Model>(model, t);
        }

        double enof_targets() {
            double res = 0;
            for(auto& t : lmb.targets.targets) {
                res += t->r;
            }
            return res;
        }

        unsigned nof_targets(const double r_lim) {
            unsigned res = 0;
            for(auto& t : lmb.targets.targets) {
                if(t->r >= r_lim) { ++res; }
            }
            return res;
        }

        Summaries get_targets() {
            Summaries res;
            for(auto& t : lmb.targets.targets) {
                res.emplace_back(t->pdf.mean(), t->pdf.cov(), t->r, t->id);
            }
            return res;
        }
};


template<typename T>
std::string print(const T& o) {
    std::stringstream s;
    o.repr(s);
    return s.str();
}


PYBIND11_MODULE(lmb, m) {
    typedef PositionSensor<Tracker::Target> PosSensor;
    py::class_<PosSensor>(m, "PositionSensor")
        .def(py::init())
        .def_readwrite("lambdaB", &PosSensor::lambdaB)
        .def_readwrite("pD", &PosSensor::pD_)
        .def("set_fov", &PosSensor::set_fov);
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
        .def(py::init<double, Tracker::Gaussian::State, Tracker::Gaussian::Covariance>())
        .def_readwrite("w", &Tracker::Gaussian::w)
        .def_readwrite("m", &Tracker::Gaussian::m)
        .def_readwrite("P", &Tracker::Gaussian::P)
        .def("__repr__", &print<Tracker::Gaussian>);
    py::class_<TargetSummary>(m, "Target")
        .def(py::init())
        .def_readonly("m", &TargetSummary::m)
        .def_readonly("P", &TargetSummary::P)
        .def_readonly("r", &TargetSummary::r)
        .def_readonly("id", &TargetSummary::id)
        .def("__repr__", &print<TargetSummary>);
    py::class_<GaussianReport>(m, "GaussianReport")
        .def(py::init<GaussianReport::Measurement, GaussianReport::Covariance, double>())
        .def_readwrite("z", &GaussianReport::z)
        .def_readwrite("R", &GaussianReport::R)
        .def_readwrite("kappa", &GaussianReport::kappa)
        .def("__repr__", &print<GaussianReport>);
    py::class_<Tracker::Params>(m, "Params")
        .def_readwrite("w_lim", &Tracker::Params::w_lim)
        .def_readwrite("r_lim", &Tracker::Params::r_lim)
        .def_readwrite("nhyp_max", &Tracker::Params::nhyp_max)
        .def_readwrite("rB_max", &Tracker::Params::rB_max)
        .def_readwrite("nstd", &Tracker::Params::nstd);
    py::class_<PyInterface>(m, "Tracker")
        .def(py::init())
        .def_readwrite("params", &PyInterface::params)
        .def("correct", &PyInterface::template correct<PosSensor>, py::call_guard<py::gil_scoped_release>())
        .def("predict", (void (PyInterface::*)(NM<Tracker::Target>&, double)) &PyInterface::predict<NM<Tracker::Target>>, py::call_guard<py::gil_scoped_release>())
        .def("predict", (void (PyInterface::*)(CV<Tracker::Target>&, double)) &PyInterface::predict<CV<Tracker::Target>>, py::call_guard<py::gil_scoped_release>())
        .def("enof_targets", &PyInterface::enof_targets)
        .def("nof_targets", &PyInterface::nof_targets)
        .def("get_targets", &PyInterface::get_targets);
}
