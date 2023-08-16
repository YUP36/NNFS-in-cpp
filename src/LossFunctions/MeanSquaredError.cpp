#include "../../include/LossFunctions/MeanSquaredError.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

MeanSquaredError::MeanSquaredError() {
    dinputs = nullptr;
}

VectorXd MeanSquaredError::forward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    MatrixXd squaredError = (*yTrue - *yPredicted).array().square();
    return squaredError.rowwise().mean();
}

void MeanSquaredError::backward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    int numSamples = yPredicted->rows();
    int numOutputs = yPredicted->cols();

    if(!dinputs) dinputs = new MatrixXd(numSamples, numOutputs);
    *dinputs = -2 * (*yTrue - *yPredicted) / (numSamples * numOutputs);
}

MatrixXd* MeanSquaredError::getDinputs() const {
    return dinputs;
}
