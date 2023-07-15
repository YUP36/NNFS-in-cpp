#include "../include/ActivationSoftmax.h"

ActivationSoftmax::ActivationSoftmax() {}

void ActivationSoftmax::forward(Eigen::MatrixXd input) {
    Eigen::MatrixXd expInput = input.array().exp();
    Eigen::MatrixXd sums = expInput.rowwise().sum();
    output = expInput.array() / sums.replicate(1, input.cols()).array();
}

Eigen::MatrixXd ActivationSoftmax::getOutput() const {
    return output;
}

