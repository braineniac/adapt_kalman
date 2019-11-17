#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

class KalmanFilter {
public:
    KalmanFilter(MatrixXd, MatrixXd, MatrixXd,
                 MatrixXd, MatrixXd, MatrixXd,
                 MatrixXd, MatrixXd);

    MatrixXd R_k, Q_k;
    VectorXd x_k_pre, x_k_post, x_k_extr;
    VectorXd x0;
    MatrixXd P_k_pre, P_k_post, P_k_extr;

    VectorXd u_k, y_k;
    MatrixXd L_k;

    MatrixXd Phi_k;
    MatrixXd Gamma_k;
    MatrixXd C_k;
    MatrixXd D_k;

    MatrixXd G_k;
    MatrixXd H_k;

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

int main() {
    Eigen::Matrix2f Q;
    Q << Eigen::Matrix2f::Zero();
    std::cout << Q.size() << std::endl;
}
