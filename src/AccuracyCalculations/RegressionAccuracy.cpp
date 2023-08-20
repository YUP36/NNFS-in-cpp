#include "../../include/AccuracyCalculations/RegressionAccuracy.h"

using Eigen::MatrixXd;

RegressionAccuracy::RegressionAccuracy() {
    precision = 0.0;
}

void RegressionAccuracy::initialize(MatrixXd* yTrue, bool reinit) {
    if((precision == 0.0) || reinit) {
        double standardDeviation = (yTrue->array() - yTrue->mean()).square().sum() / yTrue->rows();
        precision = standardDeviation / 250.0;
    }
}

MatrixXd RegressionAccuracy::compare(MatrixXd* predictions, MatrixXd* yTrue) {
    double p = precision;
    MatrixXd difference = (*predictions - *yTrue).array().abs();
    return difference.unaryExpr([p](double x){return (x < p) ? 1.0 : 0.0;});
}
