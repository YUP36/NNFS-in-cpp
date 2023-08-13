#ifndef SGD_H
#define SGD_H

#include <Eigen/Dense>
#include "../Layers/Dense.h"

class SGD {

    public:
        SGD(double lr = 1.0, double dr = 0.0, double m = 0.0);
        double getLearningRate();
        void decay();
        void updateParameters(Dense* layer);
        void incrementIteration();

    private:
        double learningRate;
        double currentLearningRate;
        double decayRate;
        double iteration;
        double momentum;

};

#endif