// Copyright 2018 Jonatan Olofsson
#pragma once
#include <Eigen/Core>
#include <cmath>

namespace cf {

using LLA = Eigen::Vector3d;
using NED = Eigen::Vector3d;
using ECEF = Eigen::Vector3d;

inline double sind(const double deg) { return std::sin(deg * M_PI / 180.0); }
inline double cosd(const double deg) { return std::cos(deg * M_PI / 180.0); }

template<typename LLA_ORIGIN>
Eigen::Matrix3d nedrot(const LLA_ORIGIN& origin) {
    double lat, lon, slat, slon, clat, clon;
    Eigen::Matrix3d R;

    lat = origin(0);
    lon = origin(1);

    slat = sind(lat);
    slon = sind(lon);
    clat = cosd(lat);
    clon = cosd(lon);

    R << -slat * clon,  -slat * slon,   clat,
         -slon,         clon,           0,
         -clat * clon,  -clat * slon,   -slat;
    return R;
}

template<typename LLA>
Eigen::Vector3d lla2ecef(const LLA& lla) {
    static const double a = 6378137.0;               // WGS-84 semi-major axis
    static const double e2 = 6.6943799901377997e-3;  // WGS-84 first eccentricity squared        // NOLINT

    Eigen::Vector3d ecef;
    double lat, lon, alt, n, nalt, slat, slat2, slon, clon, clat;
    lat = lla(0);
    lon = lla(1);
    alt = lla(2);

    slat = sind(lat);
    slon = sind(lon);
    clat = cosd(lat);
    clon = cosd(lon);
    slat2 = slat * slat;

    n = a / std::sqrt(1 - e2 * slat2);
    nalt = n + alt;


    ecef << nalt * clat * clon,
            nalt * clat * slon,
            (n * (1 - e2) + alt) * slat;

    return ecef;
}

template<typename ECEF, typename LLA_ORIGIN>
Eigen::Vector3d ecef2ned(const ECEF& ecef, const LLA_ORIGIN& origin) {
    auto ecef0 = lla2ecef(origin);
    return nedrot(origin) * (ecef - ecef0);
}

template<typename LLA, typename LLA_ORIGIN>
Eigen::Vector3d lla2ned(const LLA& lla, const LLA_ORIGIN& origin) {
    auto ecef = lla2ecef(lla);
    return ecef2ned(ecef, origin);
}

template<typename ECEF>
Eigen::Vector3d ecef2lla(const ECEF& ecef) {
    // See: http://danceswithcode.net/engineeringnotes/geodetic_to_ecef/geodetic_to_ecef.html
    // Olson, 1996

    Eigen::Vector3d lla;

    static const double a = 6378137.0;               // WGS-84 semi-major axis
    static const double e2 = 6.6943799901377997e-3;  // WGS-84 first eccentricity squared        // NOLINT
    static const double a1 = 4.2697672707157535e+4;  // a1 = a*e2
    static const double a2 = 1.8230912546075455e+9;  // a2 = a1*a1
    static const double a3 = 1.4291722289812413e+2;  // a3 = a1*e2/2
    static const double a4 = 4.5577281365188637e+9;  // a4 = 2.5*a2
    static const double a5 = 4.2840589930055659e+4;  // a5 = a1+a3
    static const double a6 = 9.9330562000986220e-1;  // a6 = 1-e2

    double x, y, z, zp, w2, w, z2, r2, r, s2, c2, s, c, ss;
    double g, rg, rf, u, v, m, f, p, lat, lon, alt;

    x = ecef(0);
    y = ecef(1);
    z = ecef(2);

    zp = std::abs(z);
    w2 = x * x + y * y;
    w = std::sqrt(w2);
    z2 = z * z;
    r2 = w2 + z2;
    r = std::sqrt(r2);

    lon = std::atan2(y, x);
    s2 = z2 / r2;
    c2 = w2 / r2;
    u = a2 / r;
    v = a3 - a4 / r;
    if (c2 > .3) {
        s = (zp / r) * (1. + c2 * (a1 + u + s2 * v) / r);
        lat = std::asin(s);
        ss = s * s;
        c = std::sqrt(1. - ss);
    } else {
        c = (w / r) * (1. - s2 * (a5 - u - c2 * v) / r);
        lat = std::acos(c);
        ss = 1. - c * c;
        s = std::sqrt(ss);
    }
    g = 1. - e2 * ss;
    rg = a / sqrt(g);
    rf = a6 * rg;
    u = w - rg * c;
    v = zp - rf * s;
    f = c * u + s * v;
    m = c * v - s * u;
    p = m / (rf / g + f);
    lat += p;
    alt = f + m * p / 2;
    if (z < 0.) {
        lat = -lat;
    }
    lla << lat * 180.0 / M_PI,
           lon * 180.0 / M_PI,
           alt;
    return lla;
}

template<typename NED, typename LLA_ORIGIN>
Eigen::Vector3d ned2ecef(const NED& ned, const LLA_ORIGIN& origin) {
    auto ecef0 = lla2ecef(origin);
    return ecef0 + nedrot(origin).transpose() * ned;
}

template<typename NED, typename LLA_ORIGIN>
Eigen::Vector3d ned2lla(const NED& ned, const LLA_ORIGIN& origin) {
    auto ecef = ned2ecef(ned, origin);
    return ecef2lla(ecef);
}


}  // namespace cf
