#ifndef CATEGORICALACCURACY_H
#define CATEGORICALACCURACY_H

#include <Eigen/Dense>

#include "../ModelWrappers/Accuracy.h"

class CategoricalAccuracy : public Accuracy {

    public:
        CategoricalAccuracy();
        Eigen::MatrixXd compare(Eigen::MatrixXd* predictions, Eigen::MatrixXd* yTrue) override;

};

#endif