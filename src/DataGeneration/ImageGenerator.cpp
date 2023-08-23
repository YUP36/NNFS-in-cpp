#include <iostream>
#include <vector>
#include "../../include/DataGeneration/lodepng.h"
#include "../../include/DataGeneration/ImageGenerator.h"

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



    //////////////////////////////////////////////////////////////////////////
    ///////////////////////////// VISUALIZATION //////////////////////////////
    // const int WIDTH = 1000;
    // const int HEIGHT = 1000;
    // MatrixXd inputGrid = MatrixXd(WIDTH * HEIGHT, 2);
    // for(int y = 0; y < HEIGHT; y++) {
    //     for(int x = 0; x < WIDTH; x++) {
    //         inputGrid((y * HEIGHT) + x, 0) = (x * 2.0 / WIDTH) - 1;
    //         inputGrid((y * HEIGHT) + x, 1) = (y * 2.0 / HEIGHT) - 1;
    //     }
    // }

    // dense1.forward(&inputGrid);
    // activation1.forward(dense1.getOutput());
    // dense2.forward(activation1.getOutput());
    // activation2.forward(dense2.getOutput());

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


    // //////////////////////////////////////////////////////////////////////////
    // ///////////////////////////// VISUALIZATION //////////////////////////////
    // const int WIDTH = 1000;
    // const int HEIGHT = 1000;
    // MatrixXd inputGrid = MatrixXd(WIDTH * HEIGHT, 2);
    // for(int y = 0; y < HEIGHT; y++) {
    //     for(int x = 0; x < WIDTH; x++) {
    //         inputGrid((y * HEIGHT) + x, 0) = (x * 2.0 / WIDTH) - 1;
    //         inputGrid((y * HEIGHT) + x, 1) = (y * 2.0 / HEIGHT) - 1;
    //     }
    // }

    // Dense1.forward(&inputGrid);
    // activation1.forward(Dense1.getOutput());
    // Dense2.forward(activation1.getOutput());
    // activationLoss.forward(Dense2.getOutput());

    // std::vector<unsigned char> pixels(WIDTH * HEIGHT * 4); // RGBA format
    // RowVectorXd pix = RowVectorXd(1, 3);
    // for (int y = 0; y < HEIGHT; ++y) {
    //     for (int x = 0; x < WIDTH; ++x) {
    //         int index = 4 * (y * WIDTH + x);
    //         pix = activationLoss.getOutput()->row(y * HEIGHT + x);
    //         // green: 109, 209, 129
    //         // red: 255, 130, 130
    //         // blue: 72, 133, 232
    //         pixels[index + 0] = sqrt(109 * 109 * pix(0) + 255 * 255 * pix(1) + 72 * 72 * pix(2));
    //         pixels[index + 1] = sqrt(209 * 209 * pix(0) + 130 * 130 * pix(1) + 133 * 133 * pix(2));
    //         pixels[index + 2] = sqrt(129 * 129 * pix(0) + 130 * 130 * pix(1) + 232 * 232 * pix(2));
    //         pixels[index + 3] = 255; // Alpha channel (opacity: 255 = fully opaque)
    //         // pixels[4 * (y * WIDTH + x) + gridPredictions(HEIGHT * y + x, 0)] = 255;
    //     }
    // }
    // ImageGenerator gen = ImageGenerator();
    // // gen.createImage(pixels, "visualizations/adam/512lr0.02dr1e-5wrdo0.1.png", WIDTH, HEIGHT);
