#include "../../include/ActivationFunctions/Linear.h"

using Eigen::MatrixXd;

Linear::Linear() {
    output = nullptr;
    dinputs = nullptr;
}

std::string Linear::getName() const {
    return "Linear";
}

void Linear::forward(MatrixXd* in) {
    if(!output || (in->rows() != output->rows())) output = new MatrixXd(in->rows(), in->cols());
    *output = *in;
}

MatrixXd* Linear::getOutput() const {
    return output;
}

MatrixXd Linear::getPredictions() const {
    return *output;
}

void Linear::backward(MatrixXd* dvalues) {
    if(!dinputs || (dvalues->rows() != dinputs->rows())) dinputs = new MatrixXd(dvalues->rows(), dvalues->cols());
    *dinputs = *dvalues;
}

MatrixXd* Linear::getDinputs() const {
    return dinputs;
}
