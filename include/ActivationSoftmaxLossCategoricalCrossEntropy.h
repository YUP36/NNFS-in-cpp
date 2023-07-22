#ifndef ACTIVATIONSOFTMAXLOSSCATEGORICALCROSSENTROPY_H
#define ACTIVATIONSOFTMAXLOSSCATEGORICALCROSSENTROPY_H

#include "../include/ActivationSoftmax.h"
#include "../include/LossCategoricalCrossEntropy.h"
#include <Eigen/Dense>

class ActivationSoftmaxLossCategoricalCrossEntropy {
    
    public:
        ActivationSoftmaxLossCategoricalCrossEntropy();
        Eigen::MatrixXd getDinputs();
        double forwardAndCalculate(Eigen::MatrixXd inputs, Eigen::VectorXi yTrue);
        void backward(Eigen::MatrixXd yPredicted, Eigen::VectorXi yTrue);

    private:
        ActivationSoftmax activation;
        LossCategoricalCrossEntropy loss;
        Eigen::MatrixXd dinputs;

};

#endif