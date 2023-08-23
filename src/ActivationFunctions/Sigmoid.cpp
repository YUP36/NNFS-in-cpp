#include "../../include/ActivationFunctions/Sigmoid.h"

#include <iostream>
using Eigen::MatrixXd;

Sigmoid::Sigmoid() {
    output = nullptr;
    dinputs = nullptr;
}

std::string Sigmoid::getName() const {
    return "Sigmoid";
}

void Sigmoid::forward(MatrixXd* in) {
    if(!output || (in->rows() != output->rows())) output = new MatrixXd(in->rows(), in->cols());
    *output = 1 / (1 + (-in->array()).exp());
}

MatrixXd* Sigmoid::getOutput() const {
    return output;
}

MatrixXd Sigmoid::getPredictions() const {
    return output->unaryExpr([](double x){return (x > 0.5) ? 1.0 : 0.0;});
}

void Sigmoid::backward(MatrixXd* dvalues) {
    if(!dinputs || (dvalues->rows() != dinputs->rows())) dinputs = new MatrixXd(dvalues->rows(), dvalues->cols());
    *dinputs = dvalues->array() * (1 - output->array()) * output->array();
}

MatrixXd* Sigmoid::getDinputs() const {
    return dinputs;
}