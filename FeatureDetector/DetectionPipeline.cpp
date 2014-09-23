//
//  DetectionPipeline.cpp
//  FeatureDetector
//
//  Created by Akul Penugonda on 9/22/14.
//  Copyright (c) 2014 Akul Penugonda. All rights reserved.
//

#include "DetectionPipeline.h"

void DetectionPipeline::runPipeline(char* image) {
    //Read image from file
    Mat im = imread(image, 1);
    
    //Perform Keypoint Finding
    
    int minHessian = 400;
    SiftFeatureDetector detector(minHessian);
    
    vector<KeyPoint> imageKeypoints;
    
    detector.detect(im, imageKeypoints);
    
    //Draw Keypoints on Image
    
    Mat keypointsImage;
    
    drawKeypoints(im, imageKeypoints, keypointsImage, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    imshow("Keypoints", keypointsImage);
    
    char nameC[1000];
    strcpy(nameC, objectOutput);
    strcat(nameC, "/");
    strcat(nameC, objType);
    strcat(nameC, "/");
    strcat(nameC, imageName);
    printf("File Stored: %s\n", nameC);
    
    //Write image to file
    
    imwrite(nameC, keypointsImage);
}

void DetectionPipeline::setDir(char* dir) {
    directory = dir;
}

void DetectionPipeline::setObjType(char* type) {
    objType = type;
}

void DetectionPipeline::setImageName(char *name) {
    imageName = name;
}