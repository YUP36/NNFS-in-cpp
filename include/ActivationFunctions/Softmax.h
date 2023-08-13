#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <Eigen/Dense>

class Softmax {

    public:
        Softmax();
        Eigen::MatrixXd* getOutput() const;
        Eigen::MatrixXd* getDinputs() const;
        void forward(Eigen::MatrixXd* input);
        void backward(Eigen::MatrixXd* dvalues);
        
    private:
        Eigen::MatrixXd* dinputs;
        Eigen::MatrixXd* output;

};

#endif