#ifndef LOSSCATEGORICALCROSSENTROPY_H
#define LOSSCATEGORICALCROSSENTROPY_H

#include <iostream>
#include "../include/Loss.h"
#include <Eigen/Dense>

class LossCategoricalCrossEntropy : public Loss {

    public:
        LossCategoricalCrossEntropy();
        Eigen::VectorXd forward(Eigen::MatrixXd, Eigen::VectorXi) override;
};

#endif