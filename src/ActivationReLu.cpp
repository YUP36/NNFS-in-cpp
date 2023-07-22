#include "../include/ActivationReLu.h"

ActivationReLu::ActivationReLu() {}

Eigen::MatrixXd ActivationReLu::getOutput() const {
    return output;
}

void ActivationReLu::forward(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
    output = this->inputs.unaryExpr([](double x){return std::max(0.0, x);});
}

void ActivationReLu::backward(Eigen::MatrixXd dvalues) {
    dinputs = dvalues;
    for(int i = 0; i < inputs.rows(); i++) {
        for(int j = 0; j < inputs.cols(); j++) {
            if(inputs(i, j) < 0) {
                dinputs(i, j) = 0;
            }
        }
    }
    // dinputs = dvalues.cwiseProduct(inputs.unaryExpr([](double x){return (x > 0) ? 1 : 0;}));
}
