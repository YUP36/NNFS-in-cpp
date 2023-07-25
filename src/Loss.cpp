#include "../include/Loss.h"

using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::VectorXd;

Loss::Loss() {}

double Loss::calculate(MatrixXd* yPredicted, VectorXi* yTrue) {
    VectorXd losses = forward(yPredicted, yTrue);
    return losses.mean();
}

VectorXd Loss::forward(MatrixXd* yPredicted, VectorXi* yTrue) {
    return VectorXd::Zero(yPredicted->rows());
}
