#ifndef MEANSQUAREDERROR_H
#define MEANSQUAREDERROR_H

#include <Eigen/Dense>
#include "Loss.h"

class MeanSquaredError : public Loss {

    public:
        MeanSquaredError();
        Eigen::VectorXd forward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue) override;
        void backward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue);
        Eigen::MatrixXd* getDinputs() const;

    private:
        Eigen::MatrixXd* dinputs;
};

#endif