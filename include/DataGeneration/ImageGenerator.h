#ifndef IMAGEGENERATOR_H
#define IMAGEGENERATOR_H

#include <vector>

class ImageGenerator {

    public:
        ImageGenerator();
        static void createImage(const std::vector<unsigned char>& pixels, const std::string& filename, const int WIDTH, const int HEGIHT);
};

#endif