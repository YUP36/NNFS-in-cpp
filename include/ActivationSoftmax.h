#ifndef ACTIVATIONSOFTMAX_H
#define ACTIVATIONSOFTMAX_H

#include <iostream>
#include <Eigen/Dense>

class ActivationSoftmax {

    public:
        ActivationSoftmax();
        Eigen::MatrixXd getOutput() const;
        Eigen::MatrixXd getDinputs() const;
        void forward(Eigen::MatrixXd inputs);
        void backward(Eigen::MatrixXd dvalues);

    private:
        int numOutputs;
        int numSamples;
        Eigen::MatrixXd output;
        Eigen::MatrixXd dinputs;

};

#endif