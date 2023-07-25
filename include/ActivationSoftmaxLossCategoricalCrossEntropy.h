#ifndef ACTIVATIONSOFTMAXLOSSCATEGORICALCROSSENTROPY_H
#define ACTIVATIONSOFTMAXLOSSCATEGORICALCROSSENTROPY_H

#include "../include/ActivationSoftmax.h"
#include "../include/LossCategoricalCrossEntropy.h"
#include <Eigen/Dense>

class ActivationSoftmaxLossCategoricalCrossEntropy {
    
    public:
        ActivationSoftmaxLossCategoricalCrossEntropy();
        double forwardAndCalculate(Eigen::MatrixXd* inputs, Eigen::VectorXi* yTrue);
        Eigen::MatrixXd* getOutput() const;
        Eigen::MatrixXd* getDinputs() const;
        void backward(Eigen::MatrixXd* yPredicted, Eigen::VectorXi* yTrue);

    private:
        ActivationSoftmax activation;
        LossCategoricalCrossEntropy loss;
        Eigen::MatrixXd* dinputs;

};

#endif