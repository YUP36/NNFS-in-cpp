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
    MatrixXd weightUpdate = (momentum * *layer->getWeightMomentum()) - (currentLearningRate * *layer->getDweights());
    RowVectorXd biasUpdate = (momentum * *layer->getBiasMomentum()) - (currentLearningRate * *layer->getDbiases());

    layer->setWeightMomentum(&weightUpdate);
    layer->setBiasMomentum(&biasUpdate);

    layer->updateWeights(&weightUpdate);
    layer->updateBiases(&biasUpdate);
}

void OptimizerSGD::incrementIteration() {
    iteration++;
}