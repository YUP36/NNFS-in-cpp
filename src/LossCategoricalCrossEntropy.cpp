#include "../include/LossCategoricalCrossEntropy.h"

using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::VectorXd;
using Eigen::placeholders::all;

LossCategoricalCrossEntropy::LossCategoricalCrossEntropy() {}

MatrixXd LossCategoricalCrossEntropy::getDinputs() const {
    return dinputs;
}

VectorXd LossCategoricalCrossEntropy::forward(MatrixXd yPredicted, VectorXi yTrue) {
    int numSamples = yTrue.rows();
    VectorXd output = VectorXd::Zero(numSamples);
    MatrixXd yClipped = yPredicted.unaryExpr([](double x){return std::max(std::min(x, 1-1e-7), 1e-7);});
    
    for(int rowIndex = 0; rowIndex < numSamples; rowIndex++) {
        output(rowIndex) = yClipped(rowIndex, yTrue(rowIndex));
    }
    output = (-1 * output.array().log()).eval();

    return output;
}

void LossCategoricalCrossEntropy::backward(MatrixXd yPredictions, VectorXi yTrue) {
    int numLabels = yPredictions.cols();
    MatrixXd identity = MatrixXd::Identity(numLabels, numLabels);
    MatrixXd oneHotYTrue = identity(all, yTrue);
    dinputs = -1 * (oneHotYTrue.array() / yPredictions.array());
}
