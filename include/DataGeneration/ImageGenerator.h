#ifndef IMAGEGENERATOR_H
#define IMAGEGENERATOR_H

#include <vector>

class ImageGenerator {

    public:
        ImageGenerator();
        static void encodeImage(const std::vector<unsigned char>& pixels, const std::string& filename, const int WIDTH, const int HEGIHT);
        static std::vector<unsigned char> decodeImage(const char* filename);

};

#endif