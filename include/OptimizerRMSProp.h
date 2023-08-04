#ifndef OPTIMIZERRMSPROP_H
#define OPTIMIZERRMSPROP_H

#include <Eigen/Dense>
#include "DenseLayer.h"

class OptimizerRMSProp {

    public:
        OptimizerRMSProp(double lr = 1.0, double dr = 0.0, double e = 1e-7, double r = 0.9);
        double getLearningRate();
        void decay();
        void updateParameters(DenseLayer* layer);
        void incrementIteration();

    private:
        double learningRate;
        double currentLearningRate;
        double decayRate;
        double iteration;
        double epsilon;
        double rho;

};

#endif