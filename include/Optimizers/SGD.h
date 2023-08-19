#ifndef SGD_H
#define SGD_H

#include <Eigen/Dense>
#include "../Layers/Dense.h"
#include "../ModelWrappers/Optimizer.h"

class SGD : public Optimizer{

    public:
        SGD(double lr = 1.0, double dr = 0.0, double m = 0.0);
        double getLearningRate();
        void decay() override;
        void updateParameters(Dense* layer) override;
        void incrementIteration() override;

    private:
        double learningRate;
        double currentLearningRate;
        double decayRate;
        double iteration;
        double momentum;

};

#endif