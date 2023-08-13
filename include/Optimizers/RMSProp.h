#ifndef RMSPROP_H
#define RMSPROP_H

#include <Eigen/Dense>
#include "../Layers/Dense.h"

class RMSProp {

    public:
        RMSProp(double lr = 1.0, double dr = 0.0, double e = 1e-7, double r = 0.9);
        double getLearningRate();
        void decay();
        void updateParameters(Dense* layer);
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