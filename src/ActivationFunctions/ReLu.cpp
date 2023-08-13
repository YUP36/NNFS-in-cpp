#include "../../include/ActivationFunctions/ReLu.h"

using Eigen::MatrixXd;

ReLu::ReLu() {
    input = nullptr;
    output = nullptr;
    dinputs = nullptr;
}

MatrixXd* ReLu::getOutput() const {
    return output;
}

void ReLu::forward(MatrixXd* in) {
    input = in;
    if(!output) output = new MatrixXd(input->rows(), input->cols());
    *output = input->unaryExpr([](double x){return std::max(0.0, x);});
}

void ReLu::backward(MatrixXd* dvalues) {
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

MatrixXd* ReLu::getDinputs() const {
    return dinputs;
}
