#ifndef Linear_H
#define Linear_H

#include <Eigen/Dense>

class Linear {

    public:
        Linear();

        void forward(Eigen::MatrixXd* in);
        Eigen::MatrixXd* getOutput() const;

        void backward(Eigen::MatrixXd* dvalues);
        Eigen::MatrixXd* getDinputs() const;

    private:
        Eigen::MatrixXd* output;
        Eigen::MatrixXd* dinputs;

};

#endif