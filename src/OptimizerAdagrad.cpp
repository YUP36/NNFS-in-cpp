#include "../include/OptimizerAdagrad.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;

OptimizerAdagrad::OptimizerAdagrad(double lr, double dr, double e) {
    learningRate = lr;
    currentLearningRate = lr;
    decayRate = dr;
    iteration = 0;
    epsilon = e;
}

double OptimizerAdagrad::getLearningRate() {
    return currentLearningRate;
}

void OptimizerAdagrad::decay() {
    currentLearningRate = learningRate * (1 / (1 + decayRate * iteration));
}

void OptimizerAdagrad::updateParameters(DenseLayer* layer) {
    MatrixXd newWeightCache = layer->getWeightCache()->array() + layer->getDweights()->array().square();
    RowVectorXd newBiasCache = layer->getBiasCache()->array() + layer->getDbiases()->array().square();

    layer->setWeightCache(&newWeightCache);
    layer->setBiasCache(&newBiasCache);

    MatrixXd weightUpdate = (-currentLearningRate * *layer->getDweights()).array()
        / (MatrixXd::Constant(layer->getWeights()->rows(), layer->getWeights()->cols(), epsilon).array() + layer->getWeightCache()->array().sqrt());
    RowVectorXd biasUpdate = (-currentLearningRate * *layer->getDbiases()).array()
        / (RowVectorXd::Constant(layer->getBiases()->rows(), layer->getBiases()->cols(), epsilon).array() + layer->getBiasCache()->array().sqrt());

    layer->updateWeights(&weightUpdate);
    layer->updateBiases(&biasUpdate);
}

void OptimizerAdagrad::incrementIteration() {
    iteration++;
}