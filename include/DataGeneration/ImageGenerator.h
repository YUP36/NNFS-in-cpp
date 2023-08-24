#ifndef IMAGEGENERATOR_H
#define IMAGEGENERATOR_H

#include <vector>
#include <string>
#include "../../include/ModelWrappers/Model.h"

class ImageGenerator {

    public:
        ImageGenerator();
        static void encodeImage(const std::vector<unsigned char>& pixels, const std::string& filename, const int WIDTH, const int HEGIHT);
        static std::vector<unsigned char> decodeImage(const char* filename);
        static void visualize2DClassifier(Model* model, std::string file, int numLabels, bool binary, const int width, const int height);

    private:
        static double weightedSqrtMean(std::vector<int> colors, Eigen::RowVectorXd confidences);

};

#endif