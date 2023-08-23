#include "../../include/LossFunctions/MeanSquaredError.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

MeanSquaredError::MeanSquaredError() {
    output = nullptr;
    dinputs = nullptr;
}

std::string MeanSquaredError::getName() const {
    return "MeanSquaredError";
}

void MeanSquaredError::forward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    if(!output || (yTrue->rows() != output->rows())) output = new VectorXd(yTrue->rows());
    *output = (*yTrue - *yPredicted).array().square();
    *output = output->rowwise().mean();
}

VectorXd* MeanSquaredError::getOutput() {
    return output;
}

void MeanSquaredError::backward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    int numSamples = yPredicted->rows();
    int numOutputs = yPredicted->cols();

    if(!dinputs || (numSamples != dinputs->rows())) dinputs = new MatrixXd(numSamples, numOutputs);
    *dinputs = -2 * (*yTrue - *yPredicted) / (numSamples * numOutputs);
}

MatrixXd* MeanSquaredError::getDinputs() const {
    return dinputs;
}
