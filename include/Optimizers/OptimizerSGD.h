#ifndef OPTIMIZERSGD_H
#define OPTIMIZERSGD_H

#include <Eigen/Dense>
#include "../DenseLayer.h"

class OptimizerSGD {

    public:
        OptimizerSGD(double lr = 1.0, double dr = 0.0, double m = 0.0);
        double getLearningRate();
        void decay();
        void updateParameters(DenseLayer* layer);
        void incrementIteration();

    private:
        double learningRate;
        double currentLearningRate;
        double decayRate;
        double iteration;
        double momentum;

};

#endif