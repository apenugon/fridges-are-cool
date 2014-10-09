//
//  DetectionPipeline.cpp
//  FeatureDetector
//
//  Created by Akul Penugonda on 9/22/14.
//  Copyright (c) 2014 Akul Penugonda. All rights reserved.
//

#include "DetectionPipeline.h"
#include "ComparisonCheck.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

DetectionPipeline::DetectionPipeline(char* detector, char* extractor, char* matcher) {
    this->detectorName = detector;
    this->extractorName = extractor;
    this->matcherName = matcher;
}

DetectionPipeline::~DetectionPipeline() {
}

size_t DetectionPipeline::runPipeline(char* image, Mat testImage, Mat testImageDescriptor, vector<KeyPoint> testImageKeypoints, bool showImages) {
    //Read image from file
    Mat im = imread(image, 1);
    
    // Step 1: Detect Keypoints
    
    int minHessian = 400;
    //SiftFeatureDetector detector(minHessian);
    
    vector<KeyPoint> imageKeypoints;
    
    Ptr<FeatureDetector> detector = FeatureDetector::create(detectorName);
    detector->detect(im, imageKeypoints);
    
    // Step 2: Extract keypoints
    
    //SiftDescriptorExtractor extractor;
    
    Mat imageDescriptor;
    
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(extractorName);
    extractor->compute(im, imageKeypoints, imageDescriptor);
    
    // Step 3: Match Keypoints
    
    vector<vector<DMatch>> matchesImageToObject;
    vector<vector<DMatch>> matchesObjectToImage;
    
    vector<DMatch> matches;
    //matcher->match(imageDescriptor, testImageDescriptor, matchesImageToObject);
    //matcher->match(testImageDescriptor, imageDescriptor, matchesObjectToImage);
    
    /*
    //Cross-Checking
    
    for (DMatch match1 : matchesImageToObject) {
        for (DMatch match2 : matchesObjectToImage) {
            if (match1.distance == match2.distance &&
                match1.queryIdx == match2.trainIdx &&
                match1.trainIdx == match2.trainIdx) {
                matches.push_back(match1);
            }
        }
    }*/
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcherName);
    matcher->knnMatch(testImageDescriptor, imageDescriptor, matchesObjectToImage, 1);
    matcher->knnMatch(imageDescriptor, testImageDescriptor, matchesImageToObject, 1);
    
    for (vector<DMatch> topMatchesObjectToImage : matchesObjectToImage) {
        for (vector<DMatch> topMatchesImageToObject : matchesImageToObject) {
            for (DMatch topMatchObjectToImage : topMatchesObjectToImage) {
                for (DMatch topMatchImageToObject : topMatchesImageToObject) {
                    if (topMatchObjectToImage.queryIdx == topMatchImageToObject.trainIdx &&
                        topMatchImageToObject.queryIdx == topMatchObjectToImage.trainIdx) {
                        matches.push_back(topMatchObjectToImage);
                    }
                }
            }
        }
    }
    
    // Min distance checking:
    /*
    double min = 10000;
    
    for (int i = 0; i < testImageDescriptor.rows; i++) {
        double distance = matchesObjectToImage[i].distance;
        
        if (distance < min)
            min = distance;
    }
    
    for (DMatch match : matchesObjectToImage) {
        if (match.distance < 2.5 * min)
            matches.push_back(match);
    }*/
    
    // Draw/Show matches
    Mat img_matches;
    drawMatches( testImage, testImageKeypoints, im, imageKeypoints, matches, img_matches, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    imshow("Crosscheck Results", img_matches);
    
    waitKey(0);

    //imshow("Matches", img_matches);
    
    // Step 4: Find homography
    // Need at least 5 matches
    if (matches.size() > 5) {
        vector<Point2f> obj;
        vector<Point2f> scene;
    
        for (int i = 0; i < matches.size(); i++) {
            obj.push_back(testImageKeypoints[matches[i].queryIdx].pt);
            scene.push_back(imageKeypoints[matches[i].queryIdx].pt);
        }
        
        vector<unsigned char> inliersMask(matches.size());
        Mat H = findHomography(obj, scene, CV_RANSAC, 1, inliersMask);
    
        //Only keep RANSAC inliers
        vector<DMatch> inliers;
        
        for (int i = 0; i < inliersMask.size(); i++) {
            if (inliersMask[i])
                inliers.push_back(matches[i]);
        }
        
        // Draw/Show matches
        Mat img_matches;
        drawMatches( testImage, testImageKeypoints, im, imageKeypoints, inliers, img_matches, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        imshow("RANSAC", img_matches);
        waitKey(0);
        
        return inliers.size();
    }
    else {
        return 0;
    }
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