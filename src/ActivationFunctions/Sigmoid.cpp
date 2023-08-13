#include "../../include/ActivationFunctions/Sigmoid.h"

using Eigen::MatrixXd;

Sigmoid::Sigmoid() {
    output = nullptr;
    dinputs = nullptr;
}

void Sigmoid::forward(MatrixXd* in) {
    if(!output) output = new MatrixXd(in->rows(), in->cols());
    *output = 1 / (1 + (-in->array()).exp());
}

MatrixXd* Sigmoid::getOutput() const {
    return output;
}

void Sigmoid::backward(MatrixXd* dvalues) {
    if(!dinputs) dinputs = new MatrixXd(dvalues->rows(), dvalues->cols());
    *dinputs = dvalues->array() * (1 - output->array()) * output->array();
}

MatrixXd* Sigmoid::getDinputs() const {
    return dinputs;
}