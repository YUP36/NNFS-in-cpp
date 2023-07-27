#ifndef ACTIVATIONSOFTMAXLOSSCATEGORICALCROSSENTROPY_H
#define ACTIVATIONSOFTMAXLOSSCATEGORICALCROSSENTROPY_H

#include <Eigen/Dense>
#include "../include/ActivationSoftmax.h"
#include "../include/LossCategoricalCrossEntropy.h"

class ActivationSoftmaxLossCategoricalCrossEntropy {
    
    public:
        ActivationSoftmaxLossCategoricalCrossEntropy();
        void forward(Eigen::MatrixXd* inputs);
        Eigen::MatrixXd* getOutput() const;
        double calculate(Eigen::VectorXi* yTrue);
        Eigen::MatrixXd* getDinputs() const;
        void backward(Eigen::MatrixXd* yPredicted, Eigen::VectorXi* yTrue);

    private:
        ActivationSoftmax activation;
        LossCategoricalCrossEntropy loss;
        Eigen::MatrixXd* dinputs;

};

#endif