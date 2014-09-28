//
//  DetectionPipeline.cpp
//  FeatureDetector
//
//  Created by Akul Penugonda on 9/22/14.
//  Copyright (c) 2014 Akul Penugonda. All rights reserved.
//

#include "DetectionPipeline.h"

void DetectionPipeline::runPipeline(char* image, Mat testImage, Mat testImageDescriptor, vector<KeyPoint> testImageKeypoints) {
    //Read image from file
    Mat im = imread(image, 1);
    
    // Step 1: Detect Keypoints
    
    int minHessian = 400;
    SiftFeatureDetector detector(minHessian);
    
    vector<KeyPoint> imageKeypoints;
    
    detector.detect(im, imageKeypoints);
    
    // Step 2: Extract keypoints
    
    SiftDescriptorExtractor extractor;
    Mat imageDescriptor;
    
    extractor.compute(im, imageKeypoints, imageDescriptor);
    
    // Step 3: Match Keypoints
    
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(imageDescriptor, testImageDescriptor, matches);

    // Draw/Show matches
    Mat img_matches;
    drawMatches(im, imageKeypoints, testImage, testImageKeypoints, matches, img_matches, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //imshow("Matches", img_matches);
    
    // Step 4: Ransac
    
    vector<Point2f> obj;
    vector<Point2f> scene;
    
    for (int i = 0; i < matches.size(); i++)
    {
        obj.push_back(testImageKeypoints[matches[i].queryIdx].pt);
        scene.push_back(imageKeypoints[matches[i].queryIdx].pt);
    }
    
    Mat H = findHomography(obj, scene, CV_RANSAC);
    
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( im.cols, 0 );
    obj_corners[2] = cvPoint( im.cols, im.rows ); obj_corners[3] = cvPoint( 0, im.rows );
    std::vector<Point2f> scene_corners(4);
    
    perspectiveTransform( obj_corners, scene_corners, H);
    
    // Draw lines between corners
    line( img_matches, scene_corners[0] + Point2f(im.cols, 0), scene_corners[1] + Point2f( im.cols, 0), Scalar(0, 255, 0));
    line( img_matches, scene_corners[1] + Point2f(im.cols, 0), scene_corners[2] + Point2f( im.cols, 0), Scalar(0, 255, 0));
    line( img_matches, scene_corners[2] + Point2f(im.cols, 0), scene_corners[3] + Point2f( im.cols, 0), Scalar(0, 255, 0));
    line( img_matches, scene_corners[3] + Point2f(im.cols, 0), scene_corners[4] + Point2f( im.cols, 0), Scalar(0, 255, 0));
    
    imshow("Matches w/ detected object", img_matches);
    
    /*
    //Draw Keypoints on Image
    
    Mat keypointsImage;
    
    drawKeypoints(im, imageKeypoints, keypointsImage, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    */
    
    /*char nameC[1000];
    strcpy(nameC, objectOutput);
    strcat(nameC, "/");
    strcat(nameC, imageName);
    printf("File Stored: %s\n", nameC);*/
    
    
    //Write image to file
    
    //imwrite(nameC, img_matches);
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