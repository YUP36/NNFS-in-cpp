#include <iostream>
#include <Eigen/Dense>
#include "../../include/DataGeneration/lodepng.h"
#include "../../include/DataGeneration/ImageGenerator.h"

using Eigen::MatrixXd;

ImageGenerator::ImageGenerator() {}

void ImageGenerator::encodeImage(const std::vector<unsigned char>& pixels, const std::string& filename, const int WIDTH, const int HEIGHT) {
    unsigned error = lodepng::encode(filename, pixels, WIDTH, HEIGHT);
    if (error) {
        std::cerr << "Encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    } else {
        std::cout << "Image saved as " << filename << std::endl;
    }
}

std::vector<unsigned char> ImageGenerator::decodeImage(const char* filename) {
    std::vector<unsigned char> image;
    unsigned width, height;

    unsigned error = lodepng::decode(image, width, height, filename);
    if(error) std::cout << "Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    return image;
}

void ImageGenerator::visualize2DClassifier(Model* model, std::string filePath, int numLabels, bool binary, const int width, const int height) {
    MatrixXd inputGrid = MatrixXd(width * height, 2);
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            inputGrid((y * height) + x, 0) = (x * 2.0 / width) - 1;
            inputGrid((y * height) + x, 1) = (y * 2.0 / height) - 1;
        }
    }
    MatrixXd output = model->predict(&inputGrid, numLabels, 128);
    if(binary) {
        MatrixXd newOutput = MatrixXd(output.rows(), 2);
        for(int row = 0; row < output.rows(); row++) {
            newOutput(row, 0) = output(row, 0);
            newOutput(row, 1) = 1 - output(row, 0);
        }
        output = MatrixXd(output.rows(), 2);
        output = newOutput;
    }
    
    std::vector<int> redColors, greenColors, blueColors;
    for(int i = 0; i < numLabels; i++) {
        redColors.push_back(rand() % 256);
        greenColors.push_back(rand() % 256);
        blueColors.push_back(rand() % 256);
    }

    std::vector<unsigned char> pixels(width * height * 4);
    Eigen::RowVectorXd pix = Eigen::RowVectorXd(1, numLabels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int rowIndex = y * width + x;
            pix = output.row(rowIndex);
            pixels[rowIndex * 4 + 0] = weightedSqrtMean(redColors, pix);
            pixels[rowIndex * 4 + 1] = weightedSqrtMean(greenColors, pix);
            pixels[rowIndex * 4 + 2] = weightedSqrtMean(blueColors, pix);
            pixels[rowIndex * 4 + 3] = 255;
        }
    }
    ImageGenerator gen = ImageGenerator();
    gen.encodeImage(pixels, filePath, width, height);
}

double ImageGenerator::weightedSqrtMean(std::vector<int> colors, Eigen::RowVectorXd confidences) {
    int sum = 0;
    for(int i = 0; i < colors.size(); i++) {
        sum += colors[i] * colors[i] * confidences(0, i);
    }
    return sqrt(sum);
}

    // std::vector<unsigned char> pixels(WIDTH * HEIGHT * 4); // RGBA format
    // double pix;
    // for (int y = 0; y < HEIGHT; ++y) {
    //     for (int x = 0; x < WIDTH; ++x) {
    //         int index = 4 * (y * WIDTH + x);
    //         pix = (*(activation2.getOutput()))(y * HEIGHT + x, 0);
    //         // green: 109, 209, 129
    //         // red: 255, 130, 130
    //         pixels[index + 0] = sqrt(109 * 109 * pix + 255 * 255 * (1 - pix));
    //         pixels[index + 1] = sqrt(209 * 209 * pix + 130 * 130 * (1 - pix));
    //         pixels[index + 2] = sqrt(129 * 129 * pix + 130 * 130 * (1 - pix));
    //         pixels[index + 3] = 255; // Alpha channel (opacity: 255 = fully opaque)
    //     }
    // }
    // ImageGenerator gen = ImageGenerator();
    // gen.createImage(pixels, "visualizations/adam/binaryCrossEntropylr0.01dr5e-7.png", WIDTH, HEIGHT);