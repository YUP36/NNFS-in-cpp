#include "../../include/LossFunctions/ActivationSoftmaxLossCategoricalCrossEntropy.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

ActivationSoftmaxLossCategoricalCrossEntropy::ActivationSoftmaxLossCategoricalCrossEntropy() {
    activation = ActivationSoftmax();
    loss = LossCategoricalCrossEntropy();

    dinputs = nullptr;
}

void ActivationSoftmaxLossCategoricalCrossEntropy::forward(MatrixXd* inputs) {
    activation.forward(inputs);
}

MatrixXd* ActivationSoftmaxLossCategoricalCrossEntropy::getOutput() const {
    return activation.getOutput();
}

double ActivationSoftmaxLossCategoricalCrossEntropy::calculate(VectorXi* yTrue) {
    return loss.calculate(activation.getOutput(), yTrue);
}

void ActivationSoftmaxLossCategoricalCrossEntropy::backward(MatrixXd* yPredicted, VectorXi* yTrue) {
    int numSamples = yPredicted->rows();
    dinputs = yPredicted;
    for(int row = 0; row < numSamples; row++) {
        (*dinputs)(row, (*yTrue)(row, 0)) -= 1;
    }
    *dinputs /= numSamples;
}

MatrixXd* ActivationSoftmaxLossCategoricalCrossEntropy::getDinputs() const {
    return dinputs;
}