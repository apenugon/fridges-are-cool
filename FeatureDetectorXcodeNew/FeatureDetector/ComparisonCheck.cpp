//
//  ComparisonCheck.cpp
//  FeatureDetector
//
//  Created by Akul Penugonda on 10/9/14.
//  Copyright (c) 2014 Akul Penugonda. All rights reserved.
//

#include "localdefs.h"
#include "ComparisonCheck.h"
#include "DetectionPipeline.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>

using namespace cv;
using namespace std;

bool ComparisonCheck::has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
    str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

char* ComparisonCheck::subdirString(char* parent, char* name) {
    char* new_str = (char*)malloc(strlen(name) + strlen(parent) + 2);
    new_str[0] = '\0';
    strcat(new_str, parent);
    strcat(new_str, "/");
    strcat(new_str, name);
    return new_str;
}

ComparisonCheck::ComparisonCheck(char* detector, char* extractor, char* matcher) {
    this->detector = detector;
    this->extractor = extractor;
    this->matcher = matcher;
    pipeline = new DetectionPipeline(detector, extractor, matcher);
}

ComparisonCheck::~ComparisonCheck() {
    delete pipeline;
}

int ComparisonCheck::runCheck(char* imagePath, char* actual) {
    cout << "Image: " << imagePath << endl;
    //Load input image
    Mat testImage = imread(imagePath, 1);
    
    //Find keypoints for image 1
    int minHessian = 400;
    Ptr<FeatureDetector> detector = FeatureDetector::create(this->detector);
    
    
    //Detect keypoints for Test Image
    vector<KeyPoint> testImKeypoints;
    
    detector->detect(testImage, testImKeypoints);
    
    //Extract keypoint info
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(this->extractor);
    Mat testImDescriptor;
    
    extractor->compute(testImage, testImKeypoints, testImDescriptor);
    
    double max = -1;
    char* name;
    //Iterate through subdirectories
    DIR *d;
    struct dirent *dir;
    d = opendir(TRAINING_DATA);
    if (d) {
        //Iterate through each directory contained here
        char* subdirName;
        char* imageName;
        while ((dir = readdir(d)) != NULL) {
            //Don't count current/parent dir/DS_STORE
            if (!strcmp(dir->d_name, ".") || !strcmp(dir->d_name, "..") || !strcmp(dir->d_name, ".DS_Store")) {
                continue;
            }
            //printf("Loading Object %s\n", dir->d_name);
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
                //printf("Image name: %s\n", imageName);
                pipeline->setObjType(dir->d_name);
                pipeline->setImageName(dir2->d_name);
                double ssd = pipeline->runPipeline(imageName, testImage, testImDescriptor, testImKeypoints, false);
                if (ssd > max) {
                    name = dir->d_name;
                    max = ssd;
                }
                free(imageName);
            }
            free(subdirName);
        }
    }
    
    printf("Best matched file: %s \nNumber of inlier matches: %f", name, max);
    //pipeline->runPipeline(name, testImage, testImDescriptor, testImKeypoints, false);
    if (strcmp(name, actual) == 0)
        return 1;
    else
        return 0;

}
