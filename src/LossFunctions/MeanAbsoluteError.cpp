#include "../../include/LossFunctions/MeanAbsoluteError.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

MeanAbsoluteError::MeanAbsoluteError() {
    dinputs = nullptr;
}

VectorXd MeanAbsoluteError::forward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    MatrixXd squaredError = (*yTrue - *yPredicted).array().abs();
    return squaredError.rowwise().mean();
}

void MeanAbsoluteError::backward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    int numSamples = yPredicted->rows();
    int numOutputs = yPredicted->cols();

    if(!dinputs) dinputs = new MatrixXd(numSamples, numOutputs);
    MatrixXd difference = *yTrue - *yPredicted;
    *dinputs = difference.unaryExpr([](double x){return (x < 0) ? 1.0 : -1.0;}) / (numSamples * numOutputs);
}

MatrixXd* MeanAbsoluteError::getDinputs() const {
    return dinputs;
}
