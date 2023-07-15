#include "../include/ActivationReLu.h"

ActivationReLu::ActivationReLu() {}

void ActivationReLu::forward(Eigen::MatrixXd input) {
    output = input.unaryExpr([](double x){return std::max(0.0, x);});
}

Eigen::MatrixXd ActivationReLu::getOutput() const {
    return output;
}