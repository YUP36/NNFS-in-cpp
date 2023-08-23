#ifndef DATAPREPROCESSOR_H
#define DATAPREPROCESSOR_H

#include <string>
#include <Eigen/Dense>

class DataPreprocessor {
    
    public:
        DataPreprocessor();
        void loadMnistDataset(std::string path);
        Eigen::MatrixXd getX();
        Eigen::MatrixXd getY();

    private:
        Eigen::MatrixXd X;
        Eigen::MatrixXd Y;
    
};

#endif