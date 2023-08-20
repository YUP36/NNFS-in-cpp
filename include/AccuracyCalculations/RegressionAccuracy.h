#ifndef REGRESSIONACCURACY_H
#define REGRESSIONACCURACY_H

#include <Eigen/Dense>

#include "../ModelWrappers/Accuracy.h"

class RegressionAccuracy : public Accuracy {

    public:
        RegressionAccuracy();
        void initialize(Eigen::MatrixXd* yTrue, bool reinit = false) override;
        Eigen::MatrixXd compare(Eigen::MatrixXd* predictions, Eigen::MatrixXd* yTrue) override;

    private:
        double precision;

};

#endif