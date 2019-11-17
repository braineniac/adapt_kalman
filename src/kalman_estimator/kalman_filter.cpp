#include <iostream>
#include <Eigen/Dense>
#include <math.h>
using namespace Eigen;

class GarryStateSpace {
public:
    GarryStateSpace(double,double,double,double,double,double,double);

    MatrixXd Phi_k;
    MatrixXd Gamma_k;
    MatrixXd C_k;
    MatrixXd D_k;

    Vector2d u_k, y_k;
    double t;
    double dt;
};

GarryStateSpace::GarryStateSpace(
    double alpha,
    double beta,
    double mass,
    double length,
    double width,
    double micro_v,
    double micro_dpsi
) {
    Phi_k.setZero(7,7);
    Phi_k(0,0) = 1;
    Phi_k(0,2) = dt;

}

class KalmanFilter {
public:
    KalmanFilter(MatrixXd, MatrixXd, MatrixXd,
                 MatrixXd, MatrixXd, MatrixXd,
                 MatrixXd, MatrixXd);
    int filter_iter(double, VectorXd, VectorXd);

    MatrixXd Phi_k;
    MatrixXd Gamma_k;
    MatrixXd C_k;
    MatrixXd D_k;

    MatrixXd R_k, Q_k;
    VectorXd x_k_pre, x_k_post, x_k_extr;
    VectorXd x0;
    MatrixXd P_k_pre, P_k_post, P_k_extr;
    MatrixXd L_k;
    MatrixXd G_k;
    MatrixXd H_k;

    Vector2d u_k, y_k;

    double t;
    double dt;
};

KalmanFilter::KalmanFilter(
    MatrixXd Phi_k,
    MatrixXd Gamma_k,
    MatrixXd C_k,
    MatrixXd D_k,
    MatrixXd G_k,
    MatrixXd H_k,
    MatrixXd Q_k,
    MatrixXd R_k) {

        // check if square
        if (Phi_k.size() % 2 != 0 || Phi_k.rows() != Phi_k.cols()) {
            throw "Phi is not square!";
        }
        // checks n dimension
        if (Gamma_k.rows() != Phi_k.rows() || G_k.rows() != Phi_k.rows()
            || C_k.cols() != Phi_k.rows()) {
            throw "Check for the n dimensions failed!";
        }
        // checks for p dimension
        if (D_k.cols() != Gamma_k.cols()) {
            throw "Check for the p dimensions failed!";
        }
        // checks for r dimension
        if (G_k.cols() != H_k.cols() || Q_k.cols() != H_k.cols()
            || Q_k.rows() != Q_k.cols()){
            throw "Check for the r dimensions failed!";
        }
        // checks for q dimension
        if (D_k.rows() != C_k.rows() || H_k.rows() != C_k.rows()
            || R_k.rows() != C_k.rows() || R_k.cols() != R_k.cols()) {
            throw "Check for the q dimensions failed!";
        }

        int n = Phi_k.size() / 2;
        int p = Gamma_k.cols();
        int r = G_k.cols();
        int q = C_k.rows();

        x_k_pre.setZero(n); x_k_post.setZero(n); x_k_extr.setZero(n);
        x0.setZero(n);
        P_k_pre.setZero(n,n); P_k_post.setZero(n,n); P_k_extr.setZero(n,n);

        u_k.setZero(p); y_k.setZero(q);
        L_k.setZero(n,q);

        t = 0;
        dt = 0;
}

int KalmanFilter::filter_iter(double time_stamp, VectorXd u, VectorXd y) {
    // time_stamp sanity check
    if (time_stamp < t) {
        throw "Time stamp can't be smaller than filter time!";
    }
    dt = time_stamp - t;
    t = time_stamp;

    // u and y size check
    if (u.size() < Gamma_k.cols() || y.size() < C_k.rows()) {
        throw "u or y dimension is wrong!";
    }
    u_k = u;
    y_k = y;

    return 0;
}

int main() {
    Eigen::Matrix2f Q;
    Q << Eigen::Matrix2f::Zero();
    std::cout << Q.size() << std::endl;
}
