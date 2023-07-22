#include "../include/ActivationSoftmax.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;

ActivationSoftmax::ActivationSoftmax() {}

Eigen::MatrixXd ActivationSoftmax::getOutput() const {
    return output;
}

Eigen::MatrixXd ActivationSoftmax::getDinputs() const {
    return dinputs;
}

void ActivationSoftmax::forward(MatrixXd inputs) {
    int numOutputs = inputs.cols();
    int numSamples = inputs.rows();
    MatrixXd expInput = inputs.array().exp();
    MatrixXd sums = expInput.rowwise().sum();
    output = expInput.array() / sums.replicate(1, numOutputs).array();
}

void ActivationSoftmax::backward(MatrixXd dvalues) {
    dinputs = MatrixXd::Zero(numSamples, numOutputs);
    RowVectorXd sampleOutput = RowVectorXd::Zero(1, numOutputs);
    RowVectorXd sampleDvalue = RowVectorXd::Zero(1, numOutputs);
    MatrixXd outputDiag = MatrixXd::Zero(numOutputs, numOutputs);

    for(int row = 0; row < numSamples; row++) {
        sampleOutput = output.row(row);
        sampleDvalue = dvalues.row(row);
        outputDiag = sampleOutput.asDiagonal();

        dinputs.row(row) = (outputDiag - (sampleOutput.transpose() * sampleOutput)) * sampleDvalue.transpose();
    }
}
