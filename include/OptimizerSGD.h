#ifndef OPTIMIZERSGD_H
#define OPTIMIZERSGD_H

#include "../include/DenseLayer.h"
#include "Eigen/Dense"

class OptimizerSGD {

    public:
        OptimizerSGD(double lr = 1.0);
        void updateParameters(DenseLayer* layer);

    private:
        double learningRate;

};

#endif