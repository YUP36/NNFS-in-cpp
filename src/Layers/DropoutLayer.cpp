#include "../../include/Layers/DropoutLayer.h"

#include <iostream>

using Eigen::MatrixXd;

DropoutLayer::DropoutLayer(double r) {
    PRECISION = 10000;
    dropoutRate = r;
    cutoff = dropoutRate * PRECISION;

    output = nullptr;
    mask = nullptr;
    dinputs = nullptr;
}

void DropoutLayer::forward(MatrixXd* in) {
    if(!mask) mask = new MatrixXd(in->rows(), in->cols());
    *mask = MatrixXd::Constant(in->rows(), in->cols(), 1.0 / (1 - dropoutRate));
    for(int row = 0; row < in->rows(); row++) {
        for(int col = 0; col < in->cols(); col++) {
            if(std::rand() % PRECISION < cutoff) {
                (*mask)(row, col) = 0;
            }
        }
    }

    if(!output) output = new MatrixXd(in->rows(), in->cols());
    *output = in->array() * mask->array();
}

MatrixXd* DropoutLayer::getOutput() {
    return output;
}

void DropoutLayer::backward(MatrixXd* dvalues) {
    if(!dinputs) dinputs = new MatrixXd(dvalues->rows(), dvalues->cols());
    *dinputs = dvalues->array() * mask->array();
}

MatrixXd* DropoutLayer::getDinputs() {
    return dinputs;
}