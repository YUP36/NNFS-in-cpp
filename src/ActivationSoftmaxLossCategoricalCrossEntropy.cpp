#include "../include/ActivationSoftmaxLossCategoricalCrossEntropy.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

ActivationSoftmaxLossCategoricalCrossEntropy::ActivationSoftmaxLossCategoricalCrossEntropy() {
    activation = ActivationSoftmax();
    loss = LossCategoricalCrossEntropy();
}

double ActivationSoftmaxLossCategoricalCrossEntropy::forwardAndCalculate(MatrixXd inputs, VectorXi yTrue) {
    activation.forward(inputs);
    return loss.calculate(activation.getOutput(), yTrue);
}

void ActivationSoftmaxLossCategoricalCrossEntropy::backward(MatrixXd yPredicted, VectorXi yTrue) {
    int numSamples = yPredicted.cols();
    dinputs = yPredicted;
    for(int row = 0; row < numSamples; row++) {
        dinputs(row, yTrue(row, 0)) -= 1;
    }
    dinputs /= numSamples;
}
