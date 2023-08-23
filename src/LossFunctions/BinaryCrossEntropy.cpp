#include "../../include/LossFunctions/BinaryCrossEntropy.h"

using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

BinaryCrossEntropy::BinaryCrossEntropy() {
    output = nullptr;
    dinputs = nullptr;
}

std::string BinaryCrossEntropy::getName() const {
    return "BinaryCrossEntropy";
}

void BinaryCrossEntropy::forward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    if(!output || (yTrue->rows() != output->rows())) output = new VectorXd(yTrue->rows());
    ArrayXd yClipped = yPredicted->unaryExpr([](double x){return std::max(std::min(x, 1-1e-7), 1e-7);});
    *output = -(yTrue->array() * yClipped.log()) - (1 - yTrue->array()) * (1 - yClipped).log();
    *output = output->rowwise().mean();
}

VectorXd* BinaryCrossEntropy::getOutput() {
    return output;
}

void BinaryCrossEntropy::backward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    int numSamples = yPredicted->rows();
    int numOutputs = yPredicted->cols();

    ArrayXd yClipped = yPredicted->unaryExpr([](double x){return std::max(std::min(x, 1-1e-7), 1e-7);});
    if(!dinputs || (numSamples != dinputs->rows())) dinputs = new MatrixXd(numSamples, numOutputs);
    *dinputs = -((yTrue->array() / yClipped) - ((1 - yTrue->array()) / (1 - yClipped))) / (numOutputs * numSamples);
}

MatrixXd* BinaryCrossEntropy::getDinputs() const {
    return dinputs;
}