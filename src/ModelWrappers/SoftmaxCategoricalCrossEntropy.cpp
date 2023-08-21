#include "../../include/ModelWrappers/SoftmaxCategoricalCrossEntropy.h"

using Eigen::MatrixXd;

SoftmaxCategoricalCrossEntropy::SoftmaxCategoricalCrossEntropy() {
    dinputs = nullptr;
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