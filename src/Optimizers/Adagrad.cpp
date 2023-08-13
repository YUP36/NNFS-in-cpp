#include "../../include/Optimizers/Adagrad.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;

Adagrad::Adagrad(double lr, double dr, double e) {
    learningRate = lr;
    currentLearningRate = lr;
    decayRate = dr;
    iteration = 0;
    epsilon = e;
}

double Adagrad::getLearningRate() {
    return currentLearningRate;
}

void Adagrad::decay() {
    currentLearningRate = learningRate * (1 / (1 + decayRate * iteration));
}

void Adagrad::updateParameters(Dense* layer) {
    MatrixXd newWeightCache = layer->getWeightCache()->array() + layer->getDweights()->array().square();
    RowVectorXd newBiasCache = layer->getBiasCache()->array() + layer->getDbiases()->array().square();

    layer->setWeightCache(&newWeightCache);
    layer->setBiasCache(&newBiasCache);

    MatrixXd weightUpdate = (-currentLearningRate * *layer->getDweights()).array()
        / (epsilon + layer->getWeightCache()->array().sqrt());
    RowVectorXd biasUpdate = (-currentLearningRate * *layer->getDbiases()).array()
        / (epsilon + layer->getBiasCache()->array().sqrt());

    layer->updateWeights(&weightUpdate);
    layer->updateBiases(&biasUpdate);
}

void Adagrad::incrementIteration() {
    iteration++;
}