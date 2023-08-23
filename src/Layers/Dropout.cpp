#include "../../include/Layers/Dropout.h"

#include <iostream>

using Eigen::MatrixXd;

Dropout::Dropout(double r) {
    PRECISION = 10000;
    dropoutRate = r;
    cutoff = dropoutRate * PRECISION;

    output = nullptr;
    mask = nullptr;
    dinputs = nullptr;
}

std::string Dropout::getName() const {
    return "Dropout";
}

void Dropout::forward(MatrixXd* in) {
    if(!mask) mask = new MatrixXd(in->rows(), in->cols());
    // *mask = MatrixXd::Constant(in->rows(), in->cols(), 1.0 / (1 - dropoutRate));
    for(int row = 0; row < in->rows(); row++) {
        for(int col = 0; col < in->cols(); col++) {
            if(std::rand() % PRECISION < cutoff) {
                (*mask)(row, col) = 0;
            } else {
                (*mask)(row, col) = 1.0 / (1 - dropoutRate);
            }
        }
    }

    if(!output || (in->rows() != output->rows())) output = new MatrixXd(in->rows(), in->cols());
    *output = in->array() * mask->array();
}

void Dropout::forward(MatrixXd* in, bool training) {
    if(training) {
        forward(in);
    } else {
        if(!output) output = new MatrixXd(in->rows(), in->cols());
        *output = *in;
    }
}

MatrixXd* Dropout::getOutput() const {
    return output;
}

void Dropout::backward(MatrixXd* dvalues) {
    if(!dinputs || (dvalues->rows() != dinputs->rows())) dinputs = new MatrixXd(dvalues->rows(), dvalues->cols());
    *dinputs = dvalues->array() * mask->array();
}

MatrixXd* Dropout::getDinputs() const {
    return dinputs;
}