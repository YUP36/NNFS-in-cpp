#include "../../include/ActivationFunctions/ActivationSigmoid.h"

using Eigen::MatrixXd;

ActivationSigmoid::ActivationSigmoid() {
    output = nullptr;
    dinputs = nullptr;
}

void ActivationSigmoid::forward(MatrixXd* in) {
    if(!output) output = new MatrixXd(in->rows(), in->cols());
    *output = 1 / (1 + (-in->array()).exp());
}

MatrixXd* ActivationSigmoid::getOutput() const {
    return output;
}

void ActivationSigmoid::backward(MatrixXd* dvalues) {
    if(!dinputs) dinputs = new MatrixXd(dvalues->rows(), dvalues->cols());
    *dinputs = dvalues->array() * (1 - output->array()) * output->array();
}

MatrixXd* ActivationSigmoid::getDinputs() const {
    return dinputs;
}