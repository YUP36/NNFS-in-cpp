#ifndef MEANABSOLUTEERROR_H
#define MEANABSOLUTEERROR_H

#include <Eigen/Dense>
#include "Loss.h"

class MeanAbsoluteError : public Loss {

    public:
        MeanAbsoluteError();
        Eigen::VectorXd forward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue) override;
        void backward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue);
        Eigen::MatrixXd* getDinputs() const;

    private:
        Eigen::MatrixXd* dinputs;
};

#endif