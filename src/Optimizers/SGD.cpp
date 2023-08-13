#include "../../include/Optimizers/SGD.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;

SGD::SGD(double lr, double dr, double m) {
    learningRate = lr;
    currentLearningRate = lr;
    decayRate = dr;
    iteration = 0;
    momentum = m;
}

double SGD::getLearningRate() {
    return currentLearningRate;
}

void SGD::decay() {
    currentLearningRate = learningRate * (1 / (1 + decayRate * iteration));
}

void SGD::updateParameters(Dense* layer) {
    MatrixXd weightUpdate = (momentum * *layer->getWeightMomentum()) - (currentLearningRate * *layer->getDweights());
    RowVectorXd biasUpdate = (momentum * *layer->getBiasMomentum()) - (currentLearningRate * *layer->getDbiases());

    layer->setWeightMomentum(&weightUpdate);
    layer->setBiasMomentum(&biasUpdate);

    layer->updateWeights(&weightUpdate);
    layer->updateBiases(&biasUpdate);
}

void SGD::incrementIteration() {
    iteration++;
}