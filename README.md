# NNFS

In this project, I coded neural networks from scratch in C++. This code was adapted from Harrison Kinsley's and Daniel Kukiela's book, titled *Neural Networks from Scratch in Python*.

## Installation
Be sure to have Eigen installed in */usr/local/include/*, or change the include path in the Makefile. Then, fork the repository and clone it to your local computer
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY
cd NNFS-in-cpp
```

## Usage
There are 4 example scripts in main.cpp, which can be uncommented for use:
* mnist: this script trains on the fashion dataset from mnist and predicts on 2 new images found in *data/fashionImages/*
* categorical cross entropy: this script trains on the spiral dataset and generates a visualization found in *data/visualizations/examples/*
* binary cross entropy: this script trains on the spiral dataset specifically with 2 labels and generates a visualization found in *data/visualizations/examples/*
* mean squared error: this script trains on the sine dataset
\\To run the program, type run commands make all and ./nnfs.
```bash
make all
./nnfs
```

## Images
<table>
  <tr>
    <th><img src="https://github.com/chrisli36/NNFS-in-cpp/blob/main/data/visualizations/adam/512lr0.02dr1e-5wrdo0.1.png" width="500">An image from the classifying the spiral dataset (example 2)</th>
    <th><img src="https://github.com/chrisli36/NNFS-in-cpp/blob/main/data/visualizations/sgd/coolMistake.png" width="500">A cool bug I ran into when first coding these visuals</th>
  </tr>
</table>

## Notes
- This project was really fun, and I could see myself expanding past the book content in the future by implementing CNNs and RNNs.
- Reading the book took about 2 months, and coding this project took about a month, since I was completely new to C++. 
- All of the matrix manipulation is done using the Eigen linear algebra library.
- A lot of code that took one line in Python would take many more in C++ because it is strongly typed whereas Python is not (hence why I needed to create many wrapper classes).
- There are definitely some inconsistencies in the style of code because my preferences changed over time, but I tried to stick to using pointers for matrices and wordy variable names.
- I also implemented my own way of visualizing the results of the 2D classifiers, which is not explained in the book.
