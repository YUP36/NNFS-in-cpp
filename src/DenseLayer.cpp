#include "../include/DenseLayer.h"

using namespace std;
// using namespace std::chrono;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;

DenseLayer::DenseLayer(int numInputs, int numNeurons){
    input = nullptr;
    output = nullptr;

    weights = new MatrixXd(numInputs, numNeurons);
    *weights = MatrixXd::Random(numInputs, numNeurons);

    biases = new RowVectorXd(1, numNeurons);
    *biases = RowVectorXd::Zero(1, numNeurons);

    dinputs = nullptr;
    dweights = nullptr;
    dbiases = nullptr;
}

ostream& operator<<(ostream& os, const DenseLayer& layer) {
    os << "Weights:\n" << layer.getWeights() << std::endl << "Biases:\n" << layer.getBiases() << endl;
    return os;
}

MatrixXd* DenseLayer::getWeights() const {
    return weights;
}

void DenseLayer::setWeights(MatrixXd newWeights) {
    *weights = newWeights;
}

RowVectorXd* DenseLayer::getBiases() const {
    return biases;
}

void DenseLayer::setBiases(RowVectorXd newBiases) {
    *biases = newBiases;
}

void DenseLayer::forward(MatrixXd* in) {
    input = in;
    output = new MatrixXd(input->rows(), weights->cols());
    // auto start = high_resolution_clock::now();
    *output = ((*input) * (*weights)).rowwise() + (*biases);
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // std::cout << duration.count() << std::endl;
}

MatrixXd* DenseLayer::getOutput() const {
    return output;
}

void DenseLayer::backward(MatrixXd* dvalues) {
    dweights = new MatrixXd(input->rows(), dvalues->cols());
    *dweights = input->transpose() * (*dvalues); // y don't we noramlize for sample size????

    dbiases = new RowVectorXd(1, dvalues->cols());
    *dbiases = dvalues->colwise().sum();

    dinputs = new MatrixXd(dvalues->rows(), weights->rows());
    *dinputs = (*dvalues) * weights->transpose();
}

MatrixXd* DenseLayer::getDinputs() const {
    return dinputs;
}

MatrixXd* DenseLayer::getDweights() const {
    return dweights;
}

RowVectorXd* DenseLayer::getDbiases() const {
    return dbiases;
}