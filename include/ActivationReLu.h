#ifndef ACTIVATIONRELU_H
#define ACTIVATIONRELU_H

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

class ActivationReLu {

    public:
        ActivationReLu();
        void forward(Eigen::MatrixXd input);
        Eigen::MatrixXd getOutput() const;

    private:
        Eigen::MatrixXd output;

};

#endif