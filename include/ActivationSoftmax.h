#ifndef ACTIVATIONSOFTMAX_H
#define ACTIVATIONSOFTMAX_H

#include <iostream>
#include <Eigen/Dense>

class ActivationSoftmax {

    public:
        ActivationSoftmax();
        Eigen::MatrixXd* getOutput() const;
        Eigen::MatrixXd* getDinputs() const;
        void forward(Eigen::MatrixXd* input);
        void backward(Eigen::MatrixXd* dvalues);
        
    private:
        Eigen::MatrixXd* dinputs;
        Eigen::MatrixXd* output;

};

#endif