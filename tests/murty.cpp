#include <Eigen/Core>
#include <murty.hpp>
#include <chrono>

using namespace lmb;
typedef std::chrono::high_resolution_clock Clock;

//Eigen::MatrixXd MURTY_COST = (Eigen::MatrixXd(5, 10) <<
              //7, 51, 52, 87, 38, 60, 74, 66, 0, 20,
              //50, 12, 0, 64, 8, 53, 0, 46, 76, 42,
              //27, 77, 0, 18, 22, 48, 44, 13, 0, 57,
              //62, 0, 3, 8, 5, 6, 14, 0, 26, 39,
              //0, 97, 0, 5, 13, 0, 41, 31, 62, 48).finished();
Eigen::MatrixXd MURTY_COST = (Eigen::MatrixXd(10, 10) <<
              7, 51, 52, 87, 38, 60, 74, 66, 0, 20,
              50, 12, 0, 64, 8, 53, 0, 46, 76, 42,
              27, 77, 0, 18, 22, 48, 44, 13, 0, 57,
              62, 0, 3, 8, 5, 6, 14, 0, 26, 39,
              0, 97, 0, 5, 13, 0, 41, 31, 62, 48,
              79, 68, 0, 0, 15, 12, 17, 47, 35, 43,
              76, 99, 48, 27, 34, 0, 0, 0, 28, 0,
              0, 20, 9, 27, 46, 15, 84, 19, 3, 24,
              56, 10, 45, 39, 0, 93, 67, 79, 19, 38,
              27, 0, 39, 53, 46, 24, 69, 46, 23, 1).finished();


int main() {
    Murty m(MURTY_COST);
    Assignment res;
    double cost;
    unsigned n = 0;
    auto t1 = Clock::now();
    while(m.draw(res, cost)) {
        //std::cout << "[" << res.transpose() << "] " << cost << std::endl;
        ++n;
    }
    auto t2 = Clock::now();
    std::cout << "Drew " << n << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0 << " s" << std::endl;
    //for (int n = 0; n < 10; ++n) {
        //m.draw(res, cost);
        //std::cout << "res: [" << res.transpose() << "] " << cost << std::endl;
    //}
    return 0;
}
