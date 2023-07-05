#include "../include/DenseLayer.h"
#include <iostream>

DenseLayer::DenseLayer(int numInputs, int numNeurons) {
    weights = MatrixXd::Random(numInputs, numNeurons);
    biases = RowVectorXd::Zero(1, numNeurons);
}

void DenseLayer::printLayer() const {
    std::cout << "Weights:" << std::endl;
    std::cout << weights << std::endl;
    std::cout << "Biases:" << std::endl;
    std::cout << biases << std::endl;
}

MatrixXd DenseLayer::getWeights() const {
    return weights;
}

RowVectorXd DenseLayer::getBiases() const {
    return biases;
}

void DenseLayer::forward(MatrixXd inputs) {
    output = (inputs * weights).rowwise() + biases;
}

MatrixXd DenseLayer::getOutput() const {
    return output;
}
