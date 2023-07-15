#include "../include/DenseLayer.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;

DenseLayer::DenseLayer(int numInputs, int numNeurons) {
    weights = MatrixXd::Random(numInputs, numNeurons);
    biases = RowVectorXd::Zero(numNeurons);
}

ostream& operator<<(ostream& os, const DenseLayer& layer) {
    os << "Weights:\n" << layer.getWeights() << std::endl << "Biases:\n" << layer.getBiases() << endl;
    return os;
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
