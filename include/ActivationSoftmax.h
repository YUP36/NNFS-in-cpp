#ifndef ACTIVATIONSOFTMAX_H
#define ACTIVATIONSOFTMAX_H

#include <iostream>
#include <Eigen/Dense>

class ActivationSoftmax {

    public:
        ActivationSoftmax();
        void forward(Eigen::MatrixXd input);
        Eigen::MatrixXd getOutput() const;

    private:
        Eigen::MatrixXd output;

};

#endif