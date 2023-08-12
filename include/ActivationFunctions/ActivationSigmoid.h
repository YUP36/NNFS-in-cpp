#ifndef ACTIVATIONSIGMOID_H
#define ACTIVATIONSIGMOID_H

#include <Eigen/Dense>

class ActivationSigmoid {

    public: 
        ActivationSigmoid();
        void forward(Eigen::MatrixXd* in);
        Eigen::MatrixXd* getOutput() const;
        void backward(Eigen::MatrixXd* dvalues);
        Eigen::MatrixXd* getDinputs() const;
        
    private:
        Eigen::MatrixXd* output;
        Eigen::MatrixXd* dinputs;

};

#endif