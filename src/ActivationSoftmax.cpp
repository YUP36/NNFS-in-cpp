#include "../include/ActivationSoftmax.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;

ActivationSoftmax::ActivationSoftmax() {
    dinputs = nullptr;
}

void ActivationSoftmax::forward(MatrixXd* input) {
    int numOutputs = input->cols();
    MatrixXd expInput = input->array().exp();
    MatrixXd sums = expInput.rowwise().sum();

    output = new MatrixXd(input->rows(), input->cols());
    *output = expInput.array() / sums.replicate(1, numOutputs).array();
}

Eigen::MatrixXd* ActivationSoftmax::getOutput() const {
    return output;
}

void ActivationSoftmax::backward(MatrixXd* dvalues) {
    int numSamples = dvalues->rows();
    int numOutputs = dvalues->cols();
    
    dinputs = new MatrixXd(numSamples, numOutputs);
    RowVectorXd sampleOutput = RowVectorXd::Zero(1, numOutputs);
    RowVectorXd sampleDvalue = RowVectorXd::Zero(1, numOutputs);
    MatrixXd outputDiag = MatrixXd::Zero(numOutputs, numOutputs);

    for(int row = 0; row < numSamples; row++) {
        sampleOutput = output->row(row);
        sampleDvalue = dvalues->row(row);
        outputDiag = sampleOutput.asDiagonal();

        dinputs->row(row) = (outputDiag - (sampleOutput.transpose() * sampleOutput)) * sampleDvalue.transpose();
    }
}

Eigen::MatrixXd* ActivationSoftmax::getDinputs() const {
    return dinputs;
}
