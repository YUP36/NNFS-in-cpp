#include "../include/Loss.h"

using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::VectorXd;

Loss::Loss() {}

double Loss::calculate(MatrixXd* yPredicted, VectorXi* yTrue) {
    return forward(yPredicted, yTrue).mean();
}

VectorXd Loss::forward(MatrixXd* yPredicted, VectorXi* yTrue) {
    return VectorXd::Zero(yPredicted->rows());
}
