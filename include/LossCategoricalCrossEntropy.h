#ifndef LOSSCATEGORICALCROSSENTROPY_H
#define LOSSCATEGORICALCROSSENTROPY_H

#include <iostream>
#include "../include/Loss.h"
#include <Eigen/Dense>

class LossCategoricalCrossEntropy : public Loss {

    public:
        LossCategoricalCrossEntropy();
        Eigen::MatrixXd getDinputs() const;
        Eigen::VectorXd forward(Eigen::MatrixXd yPredicted, Eigen::VectorXi yTrue) override;
        void backward(Eigen::MatrixXd yPredicted, Eigen::VectorXi yTrue);

    private:
        Eigen::MatrixXd dinputs;
};

#endif