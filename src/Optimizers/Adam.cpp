#include "../../include/Optimizers/Adam.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;

Adam::Adam(double lr, double dr, double e, double b1, double b2) {
    learningRate = lr;
    currentLearningRate = lr;
    decayRate = dr;
    iteration = 0;
    epsilon = e;
    beta1 = b1;
    beta2 = b2;
}

double Adam::getLearningRate() {
    return currentLearningRate;
}

void Adam::decay() {
    currentLearningRate = learningRate * (1 / (1 + decayRate * iteration));
}

void Adam::updateParameters(Dense* layer) {
    MatrixXd weightMomentum = (beta1 * layer->getWeightMomentum()->array()) + ((1 - beta1) * layer->getDweights()->array());
    RowVectorXd biasMomentum = (beta1 * layer->getBiasMomentum()->array()) + ((1 - beta1) * layer->getDbiases()->array());

    layer->setWeightMomentum(&weightMomentum);
    layer->setBiasMomentum(&biasMomentum);

    MatrixXd weightCache = (beta2 * layer->getWeightCache()->array()) + ((1 - beta2) * layer->getDweights()->array().square());
    RowVectorXd biasCache = (beta2 * layer->getBiasCache()->array()) + ((1 - beta2) * layer->getDbiases()->array().square());

    layer->setWeightCache(&weightCache);
    layer->setBiasCache(&biasCache);


    MatrixXd weightMomentumCorrected = weightMomentum / (1 - pow(beta1, 1 + iteration));
    MatrixXd biasMomentumCorrected = biasMomentum / (1 - pow(beta1, 1 + iteration));

    MatrixXd weightCacheCorrected = weightCache / (1 - pow(beta2, 1 + iteration));
    MatrixXd biasCacheCorrected = biasCache / (1 - pow(beta2, 1 + iteration));

    MatrixXd weightUpdate = (-currentLearningRate * weightMomentumCorrected.array())
        / (epsilon + weightCacheCorrected.array().sqrt());
    RowVectorXd biasUpdate = (-currentLearningRate * biasMomentumCorrected.array())
        / (epsilon + biasCacheCorrected.array().sqrt());

    layer->updateWeights(&weightUpdate);
    layer->updateBiases(&biasUpdate);
}

void Adam::incrementIteration() {
    iteration++;
}