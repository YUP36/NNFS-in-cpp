#ifndef ADAM_H
#define ADAM_H

#include <Eigen/Dense>
#include "../Layers/Dense.h"

class Adam {

    public:
        Adam(double lr = 0.001, double dr = 0.0, double e = 1e-7, double b1 = 0.9, double b2 = 0.999);
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
        double beta1;
        double beta2;

};

#endif