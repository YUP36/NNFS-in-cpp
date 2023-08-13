#include "../../include/DataGeneration/Spiral.h"

using Eigen::MatrixX2d;
using Eigen::VectorXd;
using Eigen::seqN;
using Eigen::sin;

Spiral::Spiral(int numSamples, int numClasses) {
    int numTotal = numSamples * numClasses;
    X = MatrixX2d::Zero(numTotal, 2);
    Y = VectorXd::Zero(numTotal);

    for(int classNum = 0; classNum < numClasses; classNum++) {
        VectorXd r = VectorXd::LinSpaced(numSamples, 0, 1);
        VectorXd theta = (VectorXd::LinSpaced(numSamples, classNum * 4, (classNum + 1) * 4)
                            + (VectorXd::Random(numSamples) * 0.2));

        X(seqN(classNum * numSamples, numSamples), {0}) = r.array() * ((2.5 * theta).array().sin());
        X(seqN(classNum * numSamples, numSamples), {1}) = r.array() * ((2.5 * theta).array().cos());

        Y.middleRows(classNum * numSamples, numSamples) = VectorXd::Constant(numSamples, (double) classNum);
    }
}

std::ostream& operator<<(std::ostream& os, const Spiral& dataset) {
    os << "X (Inputs):\n" << dataset.getX() << std::endl << "Y (Labels):\n" << dataset.getY() << std::endl;
    return os;
}

MatrixX2d Spiral::getX() const {
    return X;
}

VectorXd Spiral::getY() const {
    return Y;
}

