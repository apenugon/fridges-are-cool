//
//  DetectionPipeline.cpp
//  FeatureDetector
//
//  Created by Akul Penugonda on 9/22/14.
//  Copyright (c) 2014 Akul Penugonda. All rights reserved.
//

#include "DetectionPipeline.h"

size_t DetectionPipeline::runPipeline(char* image, Mat testImage, Mat testImageDescriptor, vector<KeyPoint> testImageKeypoints, bool showImages) {
    //Read image from file
    Mat im = imread(image, 1);
    
    // Step 1: Detect Keypoints
    
    int minHessian = 400;
    //SiftFeatureDetector detector(minHessian);
    
    Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
    
    vector<KeyPoint> imageKeypoints;
    
    detector->detect(im, imageKeypoints);
    
    // Step 2: Extract keypoints
    
    //SiftDescriptorExtractor extractor;
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    
    Mat imageDescriptor;
    
    extractor->compute(im, imageKeypoints, imageDescriptor);
    
    // Step 3: Match Keypoints
    
    //BFMatcher matcher(NORM_L2);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    
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
    
    
    
    

    //imshow("Matches", img_matches);
    
    
    
    // Step 4: Find homography
    // Need at least 5 matches
    if (matches.size() > 5) {
    vector<Point2f> obj;
    vector<Point2f> scene;
    
    for (int i = 0; i < matches.size(); i++)
    {
        obj.push_back(testImageKeypoints[matches[i].queryIdx].pt);
        scene.push_back(imageKeypoints[matches[i].queryIdx].pt);
    }
        
        vector<unsigned char> inliersMask(matches.size());
    Mat H = findHomography(obj, scene, CV_FM_RANSAC, 15, inliersMask);
    
        //Only keep RANSAC inliers
        vector<DMatch> inliers;
        
        for (int i = 0; i < inliersMask.size(); i++) {
            if (inliersMask[i])
                inliers.push_back(matches[i]);
        }
        
        // Draw/Show matches
        Mat img_matches;
        drawMatches( testImage, testImageKeypoints, im, imageKeypoints, inliers, img_matches, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        if (showImages) {
            imshow("Final Matches", img_matches);
        }
        
        
        return inliers.size();
        
        /*
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( im.cols, 0 );
    obj_corners[2] = cvPoint( im.cols, im.rows ); obj_corners[3] = cvPoint( 0, im.rows );
    std::vector<Point2f> scene_corners(4);
    */
    //perspectiveTransform( obj_corners, scene_corners, H);
    
        /*
    // Draw lines between corners
    line( img_matches, scene_corners[0] + Point2f(im.cols, 0), scene_corners[1] + Point2f( im.cols, 0), Scalar(0, 255, 0));
    line( img_matches, scene_corners[1] + Point2f(im.cols, 0), scene_corners[2] + Point2f( im.cols, 0), Scalar(0, 255, 0));
    line( img_matches, scene_corners[2] + Point2f(im.cols, 0), scene_corners[3] + Point2f( im.cols, 0), Scalar(0, 255, 0));
    line( img_matches, scene_corners[3] + Point2f(im.cols, 0), scene_corners[4] + Point2f( im.cols, 0), Scalar(0, 255, 0));
    */
    //imshow("Matches w/ detected object", img_matches);
    }
    else {
        return 0;
    }
    
    /*char nameC[1000];
    strcpy(nameC, objectOutput);
    strcat(nameC, "/");
    strcat(nameC, imageName);
    printf("File Stored: %s\n", nameC);*/
    
    
    //Write image to file
    
    //imwrite(nameC, img_matches);*/
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