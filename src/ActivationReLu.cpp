#include "../include/ActivationReLu.h"

using Eigen::MatrixXd;

ActivationReLu::ActivationReLu() {
    input = nullptr;
    output = nullptr;
    dinputs = nullptr;
}

MatrixXd* ActivationReLu::getOutput() const {
    return output;
}

void ActivationReLu::forward(MatrixXd* in) {
    input = in;
    if(!output) output = new MatrixXd(input->rows(), input->cols());
    *output = input->unaryExpr([](double x){return std::max(0.0, x);});
}

void ActivationReLu::backward(MatrixXd* dvalues) {
    dinputs = dvalues;
    for(int i = 0; i < input->rows(); i++) {
        for(int j = 0; j < input->cols(); j++) {
            if((*input)(i, j) < 0) {
                (*dinputs)(i, j) = 0;
            }
        }
    }
    // dinputs = dvalues.cwiseProduct(inputs.unaryExpr([](double x){return (x > 0) ? 1 : 0;}));
}

MatrixXd* ActivationReLu::getDinputs() const {
    return dinputs;
}
