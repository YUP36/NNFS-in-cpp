#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>
#include "../Layers/Dense.h"

class Loss {

    public:
        Loss();
        double calculate(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue);
        virtual Eigen::VectorXd forward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue);
        double calculateRegularizationLoss(Dense* layer);

};

#endif