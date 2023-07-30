# content-based-image-Retrieval-system
## Project Description
The purpose of this project is to continue the process of learning how to manipulate and analyze images at a pixel level. In addition, this is the project where we will be doing matching, or pattern recognition.

The overall task for this project is, given a database of images and a target image, find images in the data with similar content. For this project we will not be using neural networks or object recognition methods. Instead, we will focus on more generic characteristics of the images such as color, texture, and their spatial layout. This will give you practice with working with different color spaces, histograms, spatial features, and texture features.

# System Requirements

    Operating System: Ubuntu
    Software: CMake, Make, Terminal, Sublime Editor

# File Structure

The following files should be kept in the same folder:

    CMakeList.txt
    main.cpp
    functions.h
    olympus (image directory)

# Execution Instructions

To run the code, execute the following commands:

    cmake .
    make

This will create the executable file which can be executed by the following command:

    ./main <Directory path> <task number>

# Tasks

   1. Baseline Matching
   2. Histogram Matching
   3. Multi-histogram Matching
   4. Texture and Color Matching

Make sure to replace <Directory path> and <task number> with the appropriate values for your use case.
# Information about Tasks
## 1. Baseline Matching
Baseline matching is a simple and direct method for comparing image similarity. It utilizes a 9x9 square region from the middle of the image as a feature vector, capturing its essential characteristics.

The baseline matching process can be outlined as follows:

1. Select Target Image: Choose an image as the reference for comparison.
2. Extract Features: Take a 9x9 square region from the center of the target image to create a smaller patch, which becomes the feature vector.
3. Compare Features: Iterate through a directory of other images, applying the same feature extraction process to generate feature vectors for each image.
4. Measure Distance: Use a distance metric (e.g., sum-of-squared-difference - SSD) to determine the similarity or dissimilarity between the target image's feature vector and each image's feature vector.
5. Match and Rank: Store the computed distances in an array or vector and sort the matches based on similarity scores. Identify the top N matches (N is predefined).

## 2. Histogram Matching

Histogram matching is a method for image comparison, focusing on color distributions. Our approach involves creating a single normalized color histogram as the feature vector for each image. This histogram records the frequency of color values present in the image.

To perform histogram matching, we follow these steps:

1. Select Target Image: Choose a target image for comparison.
2. Calculate Color Histogram: Create a normalized color histogram for the target image, considering red-green (rg) chromaticity values. Divide the color value range into bins and count the pixels in each bin.
3. Compute Histograms: Calculate color histograms for all other images in the dataset using the same bin configuration.
4. Measure Similarity: Use histogram intersection as the distance metric to assess the similarity between the target image's histogram and each image's histogram. A higher intersection value indicates a closer match.
5. Match and Rank: Store the computed distances in an array or vector and sort them in ascending order. The top N matches with the smallest distances represent the closest matches to the target image.

## 3. Multi-Histogram Matching
In this task, we use multiple color histograms to represent different image regions. The histogram intersection distance metric measures similarity between histograms, ranking images based on similarity scores. The top N matches are selected using two RGB histograms: one for the top half and another for the bottom half of the image. Distances are combined through weighted averaging for final similarity scores.

## 4. Texture and Color Matching
In this task, we combine color and texture information to create a feature vector for image matching. The feature vector includes a whole image color histogram and a whole image texture histogram. Using a distance metric that weights color and texture equally, we aim to find the top N matching images for a given target image.

For texture histogram, we can use the Sobel magnitude image or other texture analysis methods like Gabor filter responses to construct the histogram.

1. Calculate both color and texture histograms for the target image.
2. Use a distance metric that considers color and texture equally to compare these histograms with other images in the dataset.
3. Identify the top N images that exhibit the closest similarity to the target image in terms of color and texture.

To evaluate the results, compare the top N matches obtained here (using color and texture histograms) with those from tasks 2 and 3. This comparison will reveal the impact of incorporating texture information on matching results in contrast to using solely color histograms or a combination of color and spatial details.

