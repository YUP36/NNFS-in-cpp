#ifndef ADAGRAD_H
#define ADAGRAD_H

#include <Eigen/Dense>
#include "../Layers/Dense.h"

class Adagrad {

    public:
        Adagrad(double lr = 1.0, double dr = 0.0, double e = 1e-7);
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

};

#endif