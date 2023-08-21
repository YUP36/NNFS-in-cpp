#ifndef SOFTMAXCATEGORICALCROSSENTROPY_H
#define SOFTMAXCATEGORICALCROSSENTROPY_H

#include <Eigen/Dense>

class SoftmaxCategoricalCrossEntropy {
    
    public:
        SoftmaxCategoricalCrossEntropy();

        Eigen::MatrixXd* getDinputs() const;
        void backward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue);

    private:
        Eigen::MatrixXd* dinputs;

};

#endif