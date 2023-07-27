#include "../include/OptimizerSGD.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;

OptimizerSGD::OptimizerSGD(double lr, double dr, double m) {
    learningRate = lr;
    currentLearningRate = lr;
    decayRate = dr;
    iteration = 0;
    momentum = m;
}

double OptimizerSGD::getLearningRate() {
    return currentLearningRate;
}

void OptimizerSGD::decay() {
    currentLearningRate = learningRate * (1 / (1 + decayRate * iteration));
}

void OptimizerSGD::updateParameters(DenseLayer* layer) {
    layer->updateWeights(-currentLearningRate * (*layer->getDweights()));
    layer->updateBiases(-currentLearningRate * (*layer->getDbiases()));
}

void OptimizerSGD::incrementIteration() {
    iteration++;
}