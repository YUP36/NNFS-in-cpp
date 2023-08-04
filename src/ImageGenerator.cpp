#include <iostream>
#include <vector>
#include "../include/lodepng.h"
#include "../include/ImageGenerator.h"

ImageGenerator::ImageGenerator() {}

void ImageGenerator::createImage(const std::vector<unsigned char>& pixels, const std::string& filename, const int WIDTH, const int HEIGHT) {
    unsigned error = lodepng::encode(filename, pixels, WIDTH, HEIGHT);
    if (error) {
        std::cerr << "Encoder error " << error << ": " << std::endl;
    } else {
        std::cout << "Image saved as " << filename << std::endl;
    }
}