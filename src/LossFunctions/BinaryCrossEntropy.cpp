#include "../../include/LossFunctions/BinaryCrossEntropy.h"

using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

BinaryCrossEntropy::BinaryCrossEntropy() {
    dinputs = nullptr;
}

VectorXd BinaryCrossEntropy::forward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    ArrayXd yClipped = yPredicted->unaryExpr([](double x){return std::max(std::min(x, 1-1e-7), 1e-7);});
    // Pretty sure the next line should have a minus    \/
    MatrixXd losses = -(yTrue->array() * yClipped.log()) + (1 - yTrue->array()) * (1 - yClipped).log();
    return losses.rowwise().mean();
}

void BinaryCrossEntropy::backward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    int numSamples = yPredicted->rows();
    int numOutputs = yPredicted->cols();

    ArrayXd yClipped = yPredicted->unaryExpr([](double x){return std::max(std::min(x, 1-1e-7), 1e-7);});
    if(!dinputs) dinputs = new MatrixXd(numSamples, numOutputs);
    *dinputs = -(yTrue->array() / yClipped - (1 - yTrue->array()) / (1 - yClipped)) / (numOutputs * numSamples);
}

MatrixXd* BinaryCrossEntropy::getDinputs() const {
    return dinputs;
}