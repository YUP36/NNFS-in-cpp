#include "../../include/ModelWrappers/Loss.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

Loss::Loss() {
    newPass();
}

std::string Loss::getName() const {
    return "Loss";
}

double Loss::calculate(MatrixXd* yPredicted, MatrixXd* yTrue) {
    forward(yPredicted, yTrue);

    int numSamples = yPredicted->rows();
    double summedSampleLosses = (*getOutput()).sum();
    
    accumulatedDataLoss += summedSampleLosses;
    accumulatedCount += numSamples;
    return summedSampleLosses / numSamples;
}

double Loss::getAverageDataLoss() {
    return accumulatedDataLoss / accumulatedCount;
}

void Loss::newPass() {
    accumulatedDataLoss = 0;
    accumulatedCount = 0;
}

void Loss::forward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    return;
}

VectorXd* Loss::getOutput() {
    return new VectorXd();
}

void Loss::backward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue) {
    return;
}

Eigen::MatrixXd* Loss::getDinputs() const {
    return new MatrixXd();
}

double Loss::calculateRegularizationLoss(Dense* layer) {
    double regularizationLoss = 0.0;
    if(layer->getLambdaL1Weight() > 0) regularizationLoss += layer->getLambdaL1Weight() * layer->getWeights()->array().abs().sum();
    if(layer->getLambdaL1Bias() > 0) regularizationLoss += layer->getLambdaL1Bias() * layer->getWeights()->array().abs().sum();
    if(layer->getLambdaL2Weight() > 0) regularizationLoss += layer->getLambdaL2Weight() * layer->getWeights()->array().square().sum();
    if(layer->getLambdaL2Bias() > 0) regularizationLoss += layer->getLambdaL2Bias() * layer->getWeights()->array().square().sum();
    return regularizationLoss;
}
