#include "../../include/LossFunctions/SoftmaxCategoricalCrossEntropy.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

SoftmaxCategoricalCrossEntropy::SoftmaxCategoricalCrossEntropy() {
    activation = Softmax();
    loss = CategoricalCrossEntropy();

    dinputs = nullptr;
}

Softmax* SoftmaxCategoricalCrossEntropy::getActivationFunction() {
    return &activation;
}

CategoricalCrossEntropy* SoftmaxCategoricalCrossEntropy::getLossFunction() {
    return &loss;
}

void SoftmaxCategoricalCrossEntropy::forward(MatrixXd* inputs) {
    activation.forward(inputs);
}

MatrixXd* SoftmaxCategoricalCrossEntropy::getOutput() const {
    return activation.getOutput();
}

double SoftmaxCategoricalCrossEntropy::calculate(MatrixXd* yTrue) {
    return loss.calculate(activation.getOutput(), yTrue);
}

void SoftmaxCategoricalCrossEntropy::backward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    int numSamples = yPredicted->rows();
    dinputs = yPredicted;
    for(int row = 0; row < numSamples; row++) {
        (*dinputs)(row, (int) ((*yTrue)(row, 0))) -= 1;
    }
    *dinputs /= numSamples;
}

MatrixXd* SoftmaxCategoricalCrossEntropy::getDinputs() const {
    return dinputs;
}