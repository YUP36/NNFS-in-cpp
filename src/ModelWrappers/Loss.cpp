#include "../../include/ModelWrappers/Loss.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

Loss::Loss() {}

double Loss::calculate(MatrixXd* yPredicted, MatrixXd* yTrue) {
    return forward(yPredicted, yTrue).mean();
}

VectorXd Loss::forward(MatrixXd* yPredicted, MatrixXd* yTrue) {
    return VectorXd::Zero(yPredicted->rows());
}

double Loss::calculateRegularizationLoss(Dense* layer) {
    double regularizationLoss = 0.0;
    if(layer->getLambdaL1Weight() > 0) regularizationLoss += layer->getLambdaL1Weight() * layer->getWeights()->array().abs().sum();
    if(layer->getLambdaL1Bias() > 0) regularizationLoss += layer->getLambdaL1Bias() * layer->getWeights()->array().abs().sum();
    if(layer->getLambdaL2Weight() > 0) regularizationLoss += layer->getLambdaL2Weight() * layer->getWeights()->array().square().sum();
    if(layer->getLambdaL2Bias() > 0) regularizationLoss += layer->getLambdaL2Bias() * layer->getWeights()->array().square().sum();
    return regularizationLoss;
}
