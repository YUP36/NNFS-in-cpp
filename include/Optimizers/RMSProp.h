#ifndef RMSPROP_H
#define RMSPROP_H

#include <Eigen/Dense>
#include "../Layers/Dense.h"
#include "../ModelWrappers/Optimizer.h"

class RMSProp : public Optimizer{

    public:
        RMSProp(double lr = 1.0, double dr = 0.0, double e = 1e-7, double r = 0.9);
        double getLearningRate() override;
        void decay() override;
        void updateParameters(Dense* layer) override;
        void incrementIteration() override;

    private:
        double learningRate;
        double currentLearningRate;
        double decayRate;
        double iteration;
        double epsilon;
        double rho;

};

#endif