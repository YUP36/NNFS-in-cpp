#include "../include/OptimizerSGD.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;

OptimizerSGD::OptimizerSGD(double lr) {
    learningRate = lr;
}

void OptimizerSGD::updateParameters(DenseLayer* layer) {
    layer->setWeights((*layer->getWeights()) - (learningRate * (*layer->getDweights())));
    layer->setBiases((*layer->getBiases()) - (learningRate * (*layer->getDbiases())));
}

