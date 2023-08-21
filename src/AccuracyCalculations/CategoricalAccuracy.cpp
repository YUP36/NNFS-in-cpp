#include "../../include/AccuracyCalculations/CategoricalAccuracy.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

#include <iostream>

CategoricalAccuracy::CategoricalAccuracy() {
}

MatrixXd CategoricalAccuracy::compare(MatrixXd* predictions, MatrixXd* yTrue) {
    MatrixXd difference = (*predictions - *yTrue).array();
    return difference.unaryExpr([](double x){return (x == 0.0) ? 1.0 : 0.0;});
}
