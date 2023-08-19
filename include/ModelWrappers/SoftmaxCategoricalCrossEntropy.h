#ifndef SOFTMAXCATEGORICALCROSSENTROPY_H
#define SOFTMAXCATEGORICALCROSSENTROPY_H

#include <Eigen/Dense>
#include "../ActivationFunctions/Softmax.h"
#include "../LossFunctions/CategoricalCrossEntropy.h"

class SoftmaxCategoricalCrossEntropy {
    
    public:
        SoftmaxCategoricalCrossEntropy();
        Softmax* getActivationFunction();
        CategoricalCrossEntropy* getLossFunction();

        void forward(Eigen::MatrixXd* inputs);
        Eigen::MatrixXd* getOutput() const;
        double calculate(Eigen::MatrixXd* yTrue);
        Eigen::MatrixXd* getDinputs() const;
        void backward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue);

    private:
        Softmax activation;
        CategoricalCrossEntropy loss;
        Eigen::MatrixXd* dinputs;

};

#endif