#include "../../include/DataGeneration/DataPreprocessor.h"

#include <vector>
#include <iostream>
#include <filesystem>
#include "../../include/DataGeneration/ImageGenerator.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;

DataPreprocessor::DataPreprocessor() {}

void DataPreprocessor::loadMnistDataset(string path) {
    ImageGenerator gen = ImageGenerator();
    vector<vector<double>> rawImages;
    vector<double> labels;
    for(const auto & file: filesystem::directory_iterator(path)) {
        string label = file.path().u8string().substr(path.size());
        string subPath = path + label + "/";
        for(const auto & image: filesystem::directory_iterator(subPath)) {
            string imageName = image.path().u8string();
            vector<unsigned char> rawImage = gen.decodeImage(imageName.c_str());
            vector<double> grayScaleImage;
            for(int i = 0; i < rawImage.size(); i+= 4) {
                grayScaleImage.push_back((int) rawImage[i]);
            }
            rawImages.push_back(grayScaleImage);
            labels.push_back(stoi(label));
        }
    }

    Y = Eigen::Map<MatrixXd>(labels.data(), labels.size(), 1);
    X = MatrixXd(rawImages.size(), rawImages[0].size());
    for(int i = 0; i < rawImages.size(); i++) {
        X.row(i) = Eigen::Map<RowVectorXd>(rawImages[i].data(), 1, rawImages[0].size());
    }
}

MatrixXd DataPreprocessor::getX() {
    return X;
}

MatrixXd DataPreprocessor::getY() {
    return Y;
}