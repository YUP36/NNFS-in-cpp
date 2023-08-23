#include "../../include/ModelWrappers/Accuracy.h"

using Eigen::MatrixXd;

Accuracy::Accuracy() {
    newPass();
}

double Accuracy::calculateAccuracy(MatrixXd* predictions, MatrixXd* yTrue) {
    int numSamples = predictions->rows();
    double summedAccuracyScores = compare(predictions, yTrue).sum();

    accumulatedAccuracy += summedAccuracyScores;
    accumulatedCount += numSamples;
    return summedAccuracyScores / numSamples;
}

double Accuracy::getAverageAccuracy() {
    return accumulatedAccuracy / accumulatedCount;
}

void Accuracy::newPass() {
    accumulatedAccuracy = 0;
    accumulatedCount = 0;
}

void Accuracy::initialize(MatrixXd* yTrue, bool reinit) {}

MatrixXd Accuracy::compare(MatrixXd* predictions, MatrixXd* yTrue) {
    return MatrixXd();
}