#pragma once

#include <Eigen/Core>

namespace lap {
    typedef Eigen::Matrix<int, Eigen::Dynamic, 1> Assignment;
    typedef Eigen::Matrix<int, Eigen::Dynamic, 1> Slack;

    template<typename CostMatrix, typename Assignment, typename RowSlack, typename ColSlack>
    void lap(const CostMatrix& C, Assignment& x, RowSlack& u, ColSlack& v, double& total_cost) {
        int N = C.rows();
        int M = C.cols();
        int inf = 30000;
        int cnt, f, f0, h, i, i1, j, j1, j2 = 0, k, u1, u2, last = 0, low, min = 0,
                                         up;
        Eigen::RowVectorXd free(N), col(M), y(M), d(M), pred(M);

        x.setZero();
        y.setZero();

        for (i = 0; i < N; ++i) { free[i] = i; }
        for (j = 0; j < M; ++j) { col[j] = j; }

        cnt = 0;
        f = N;  // Each row is still unassigned

        do {
            k = 0;
            f0 = f;
            f = 0;
            while (k < f0) {
                i = free[k];
                k = k + 1;
                u1 = C(i, 0) - v[0];
                j1 = 0;
                u2 = inf;
                for (j = 1; j < M; ++j) {
                    h = C(i, j) - v[j];
                    if (h < u2) {
                        if (h >= u1) {
                            u2 = h;
                            j2 = j;
                        } else {
                            u2 = u1;
                            u1 = h;
                            j2 = j1;
                            j1 = j;
                        }
                    }
                }
                i1 = y[j1];
                if (u1 < u2) {
                    v[j1] = v[j1] - u2 + u1;
                } else if (i1 > 0) {
                    j1 = j2;
                    i1 = y[j1];
                }
                if (i1 > 0) {
                    if (u1 < u2) {
                        k = k - 1;
                        free[k] = i1;
                        x[i1] = 0;
                    } else {
                        f = f + 1;
                        free[f] = i1;
                        x[i1] = 0;
                    }
                }
                x[i] = j1;
                y[j1] = i;
            }
            cnt = cnt + 1;
        } while (cnt < 2);
        //--------------------------------------------------------
        f0 = f;
        for (f = 0; f < f0; ++f)  //{ Find augmenting path for each unassigned row }
        {
            i1 = free[f];
            low = 0;
            up = 0;
            for (j = 0; j < M; ++j) {
                d[j] = C(i1, j) - v[j];
                pred[j] = i1;
            }
            cnt = 0;
            do {
                if (up == low)  //{ Find columns with new value for minimum d }
                {
                    last = low - 1;
                    min = d[col[up]];
                    up = up + 1;
                    for (k = up; k < M; ++k) {
                        j = col[k];
                        h = d[j];
                        if (h <= min) {
                            if (h < min) {
                                up = low;
                                min = h;
                            }
                            col[k] = col[up];
                            col[up] = j;
                            up = up + 1;
                        }
                    }
                    for (h = low; h < up - 1; ++h) {
                            j = col[h];
                            if (y[j] + cnt == 0) {
                                for (k = 0; k < last; ++k)  // { Updating of column prices }
                                {
                                    j1 = col[k];
                                    v[j1] = v[j1] + d[j1] - min;
                                }
                                do {  // { Augmentation }
                                    i = pred[j];
                                    y[j] = i;
                                    k = j;
                                    j = x[i];
                                    x[i] = k;
                                } while (i != i1);
                                cnt = 1;
                                f = f0;
                                break;
                            }
                        }
                }  //{ up=low }
                if (cnt == 0) {
                    j1 = col[low];
                    low = low + 1;
                    i = y[j1];  // { Scan a row }
                    u1 = C(i, j1) - v[j1] - min;
                    for (k = up; k < M; ++k) {
                        j = col[k];
                        h = C(i, j) - v[j] - u1;
                        if (h < d[j]) {
                            d[j] = h;
                            pred[j] = i;
                            if (h == min) {
                                if (y[j] + cnt == 0) {
                                    for (k = 0; k < last; ++k)  // { Updating of column prices }
                                    {
                                        j1 = col[k];
                                        v[j1] = v[j1] + d[j1] - min;
                                    }
                                    do {  // { Augmentation }
                                        i = pred[j];
                                        y[j] = i;
                                        k = j;
                                        j = x[i];
                                        x[i] = k;
                                    } while (i != i1);
                                    cnt = 1;
                                    f = f0;
                                } else {
                                    col[k] = col[up];
                                    col[up] = j;
                                    up = up + 1;
                                }
                            }
                        }
                    }
                }
            } while (cnt == 0);
        }  // { for f }
        total_cost = 0;
        for (i = 0; i < N; ++i) {
            j = x[i];
            u[i] = C(i, j) - v[j];
            total_cost += C(i, j);
        }
    }
}
