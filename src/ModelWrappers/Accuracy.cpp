#include "../../include/ModelWrappers/Accuracy.h"

using Eigen::MatrixXd;

Accuracy::Accuracy() {}

double Accuracy::calculateAccuracy(MatrixXd* predictions, MatrixXd* yTrue) {
    return compare(predictions, yTrue).mean();
}

void Accuracy::initialize(MatrixXd* yTrue, bool reinit) {}

MatrixXd Accuracy::compare(MatrixXd* predictions, MatrixXd* yTrue) {
    return MatrixXd();
}