#include "../../include/ActivationFunctions/ReLu.h"

using Eigen::MatrixXd;

ReLu::ReLu() {
    input = nullptr;
    output = nullptr;
    dinputs = nullptr;
}

std::string ReLu::getName() const {
    return "ReLu";
}

void ReLu::forward(MatrixXd* in) {
    if(!input) input = new MatrixXd(in->rows(), in->cols());
    *input = *in;
    if(!output) output = new MatrixXd(in->rows(), in->cols());
    *output = in->unaryExpr([](double x){return std::max(0.0, x);});
}

MatrixXd* ReLu::getOutput() const {
    return output;
}

MatrixXd ReLu::getPredictions() const {
    return *output;
}

void ReLu::backward(MatrixXd* dvalues) {
    if(!dinputs) dinputs = new MatrixXd(dvalues->rows(), dvalues->cols());
    *dinputs = *dvalues;
    for(int i = 0; i < input->rows(); i++) {
        for(int j = 0; j < input->cols(); j++) {
            if((*input)(i, j) < 0) {
                (*dinputs)(i, j) = 0;
            }
        }
    }
}

MatrixXd* ReLu::getDinputs() const {
    return dinputs;
}
