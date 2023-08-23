#ifndef ACCURACY_H
#define ACCURACY_H

#include <Eigen/Dense>

class Accuracy {

    public:
        Accuracy();
        double calculateAccuracy(Eigen::MatrixXd* predictions, Eigen::MatrixXd* yTrue);
        double getAverageAccuracy();
        void newPass();
        virtual void initialize(Eigen::MatrixXd* yTrue, bool reinit = false);
        virtual Eigen::MatrixXd compare(Eigen::MatrixXd* predictions, Eigen::MatrixXd* yTrue);
    
    private:
        double accumulatedAccuracy;
        double accumulatedCount;

};

#endif