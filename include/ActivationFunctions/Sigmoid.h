#ifndef SIGMOID_H
#define SIGMOID_H

#include <Eigen/Dense>

class Sigmoid {

    public: 
        Sigmoid();
        void forward(Eigen::MatrixXd* in);
        Eigen::MatrixXd* getOutput() const;
        void backward(Eigen::MatrixXd* dvalues);
        Eigen::MatrixXd* getDinputs() const;
        
    private:
        Eigen::MatrixXd* output;
        Eigen::MatrixXd* dinputs;

};

#endif