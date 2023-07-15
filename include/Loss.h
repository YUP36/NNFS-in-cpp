#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>

class Loss {

    public:
        Loss();
        double calculate(Eigen::MatrixXd yPredicted, Eigen::VectorXi yTrue);
        virtual Eigen::VectorXd forward(Eigen::MatrixXd yPredicted, Eigen::VectorXi yTrue);

};

#endif