#ifndef RELU_H
#define RELU_H

#include <Eigen/Dense>

class ReLu {

    public:
        ReLu();
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