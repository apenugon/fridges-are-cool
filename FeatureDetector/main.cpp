//
//  main.cpp
//  FeatureDetector
//
//  Created by Akul Penugonda on 9/22/14.
//  Copyright (c) 2014 Akul Penugonda. All rights reserved.
//

#include "DetectionPipeline.h"
#include <stdio.h>
#include <iostream>
#include <dirent.h>

#define TRAINING_DATA "/Users/apenugonda/Pictures/Objects-640"

using namespace cv;
using namespace std;

bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
    str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

char* subdirString(char* parent, char* name) {
    char* new_str = (char*)malloc(strlen(name) + strlen(parent) + 2);
    new_str[0] = '\0';
    strcat(new_str, parent);
    strcat(new_str, "/");
    strcat(new_str, name);
    return new_str;
}

int main(int argc, const char * argv[])
{
    //Load input image
    Mat testImage = imread("/Users/apenugonda/Pictures/Objects-Test-640/test-1.jpg");
    
    //Find keypoints for image 1
    int minHessian = 400;
    SiftFeatureDetector detector(minHessian);
    
    
    //Detect keypoints for Test Image
    vector<KeyPoint> testImKeypoints;
    
    detector.detect(testImage, testImKeypoints);
    
    //Extract keypoint info
    SiftDescriptorExtractor extractor;
    Mat testImDescriptor;
    
    extractor.compute(testImage, testImKeypoints, testImDescriptor);
    
    //Iterate through subdirectories
    DIR *d;
    struct dirent *dir;
    d = opendir(TRAINING_DATA);
    DetectionPipeline pipeline;
    if (d) {
        //Iterate through each directory contained here
        char* subdirName;
        char* imageName;
        while ((dir = readdir(d)) != NULL) {
            //Don't count current/parent dir/DS_STORE
            if (!strcmp(dir->d_name, ".") || !strcmp(dir->d_name, "..") || !strcmp(dir->d_name, ".DS_Store")) {
                continue;
            }
            printf("Loading Object %s\n", dir->d_name);
            subdirName = subdirString(TRAINING_DATA, dir->d_name);
            DIR *d2;
            struct dirent *dir2;
            d2 = opendir(subdirName);
            while ((dir2 = readdir(d2)) != NULL) {
                
                //Can only be a JPG file
                if (!has_suffix(dir2->d_name, ".jpg")) {
                    continue;
                }
                
                imageName = subdirString(subdirName, dir2->d_name);
                printf("Image name: %s\n", imageName);
                pipeline.setObjType(dir->d_name);
                pipeline.setImageName(dir2->d_name);
                pipeline.runPipeline(imageName, testImage, testImDescriptor, testImKeypoints);
                free(imageName);
            }
            free(subdirName);
        }
    }
    waitKey(0);
    return 0;
}

