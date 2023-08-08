#ifndef OPTIMIZERADAM_H
#define OPTIMIZERADAM_H

#include <Eigen/Dense>
#include "../DenseLayer.h"

class OptimizerAdam {

    public:
        OptimizerAdam(double lr = 1.0, double dr = 0.0, double e = 1e-7, double b1 = 0.9, double b2 = 0.999);
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
        double beta1;
        double beta2;

};

#endif