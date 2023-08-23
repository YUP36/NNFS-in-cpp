#include "../../include/LossFunctions/CategoricalCrossEntropy.h"

using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::VectorXd;
using Eigen::placeholders::all;

CategoricalCrossEntropy::CategoricalCrossEntropy() {
    output = nullptr;
    dinputs = nullptr;
}

std::string CategoricalCrossEntropy::getName() const {
    return "CategoricalCrossEntropy";
}

void CategoricalCrossEntropy::forward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    int numSamples = yTrue->rows();
    if(!output || (numSamples != output->rows())) output = new VectorXd(numSamples);

    MatrixXd yClipped = yPredicted->unaryExpr([](double x){return std::max(std::min(x, 1-1e-7), 1e-7);});
    for(int row = 0; row < numSamples; row++) {
        (*output)(row) = yClipped(row, (int) ((*yTrue)(row, 0)));
    }

    *output = -1 * output->array().log();
}

VectorXd* CategoricalCrossEntropy::getOutput() {
    return output;
}

void CategoricalCrossEntropy::backward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    int numSamples = yPredicted->rows();
    int numLabels = yPredicted->cols();

    MatrixXd identity = MatrixXd::Identity(numLabels, numLabels);
    MatrixXd oneHotYTrue = MatrixXd(numSamples, numLabels);
    for(int row = 0; row < numSamples; row++) {
        oneHotYTrue.row(row) = identity.row((int) ((*yTrue)(row, 0)));
    }
    
    if(!dinputs || (numSamples != dinputs->rows())) dinputs = new MatrixXd(numSamples, numLabels);
    *dinputs = (oneHotYTrue.array() / yPredicted->array()) / (-numSamples);
}

MatrixXd* CategoricalCrossEntropy::getDinputs() const {
    return dinputs;
}
