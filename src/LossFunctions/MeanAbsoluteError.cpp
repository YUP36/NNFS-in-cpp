#include "../../include/LossFunctions/MeanAbsoluteError.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

MeanAbsoluteError::MeanAbsoluteError() {
    output = nullptr;
    dinputs = nullptr;
}

std::string MeanAbsoluteError::getName() const {
    return "MeanAbsoluteError";
}

void MeanAbsoluteError::forward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    if(!output || (yTrue->rows() != output->rows())) output = new VectorXd(yTrue->rows());
    *output = (*yTrue - *yPredicted).array().abs();
    *output = output->rowwise().mean();
}

VectorXd* MeanAbsoluteError::getOutput() {
    return output;
}

void MeanAbsoluteError::backward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    int numSamples = yPredicted->rows();
    int numOutputs = yPredicted->cols();

    if(!dinputs || (numSamples != dinputs->rows())) dinputs = new MatrixXd(numSamples, numOutputs);
    MatrixXd difference = *yTrue - *yPredicted;
    *dinputs = difference.unaryExpr([](double x){return (x < 0) ? 1.0 : -1.0;}) / (numSamples * numOutputs);
}

MatrixXd* MeanAbsoluteError::getDinputs() const {
    return dinputs;
}
