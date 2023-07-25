#ifndef ACTIVATIONRELU_H
#define ACTIVATIONRELU_H

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

class ActivationReLu {

    public:
        ActivationReLu();
        Eigen::MatrixXd* getOutput() const;
        void forward(Eigen::MatrixXd* input);
        void backward(Eigen::MatrixXd* dvalues);
        Eigen::MatrixXd* getDinputs() const;

    private:
        Eigen::MatrixXd* input;
        Eigen::MatrixXd* output;
        Eigen::MatrixXd* dinputs;

};

#endif