#include "../../include/ActivationFunctions/Linear.h"

using Eigen::MatrixXd;

Linear::Linear() {
    output = nullptr;
    dinputs = nullptr;
}

void Linear::forward(MatrixXd* in) {
    if(!output) output = new MatrixXd(in->rows(), in->cols());
    *output = *in;
}

MatrixXd* Linear::getOutput() const {
    return output;
}

void Linear::backward(MatrixXd* dvalues) {
    if(!dinputs) dinputs = new MatrixXd(dvalues->rows(), dvalues->cols());
    *dinputs = *dvalues;
}

MatrixXd* Linear::getDinputs() const {
    return dinputs;
}
