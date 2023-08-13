#ifndef DROPOUT_H
#define DROPOUT_H

#include <Eigen/Dense>

class Dropout {
    // dropout rate only has accuracy to 5 decimal places

    public:
        Dropout(double r);
        void forward(Eigen::MatrixXd* in);
        Eigen::MatrixXd* getOutput();
        void backward(Eigen::MatrixXd* dvalues);
        Eigen::MatrixXd* getDinputs();

    private:
        double dropoutRate;
        double cutoff;
        int PRECISION;
        Eigen::MatrixXd* output;
        Eigen::MatrixXd* mask;
        Eigen::MatrixXd* dinputs;

};

#endif