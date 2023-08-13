#include "../../include/ActivationFunctions/Softmax.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;

Softmax::Softmax() {
    dinputs = nullptr;
    output = nullptr;
}

void Softmax::forward(MatrixXd* input) {
    int numOutputs = input->cols();
    MatrixXd expInput = input->array().exp();
    MatrixXd sums = expInput.rowwise().sum();

    if(!output) output = new MatrixXd(input->rows(), input->cols());
    *output = expInput.array() / sums.replicate(1, numOutputs).array();
}

Eigen::MatrixXd* Softmax::getOutput() const {
    return output;
}

void Softmax::backward(MatrixXd* dvalues) {
    int numSamples = dvalues->rows();
    int numOutputs = dvalues->cols();
    
    if(!dinputs) dinputs = new MatrixXd(numSamples, numOutputs);
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

Eigen::MatrixXd* Softmax::getDinputs() const {
    return dinputs;
}
