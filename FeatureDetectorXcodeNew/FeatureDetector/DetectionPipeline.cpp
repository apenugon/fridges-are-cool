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
#include "opencv2/video/video.hpp"
#include "opencv2/videostab/global_motion.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
#include <sys/stat.h>

using namespace std;
using namespace cv;
using namespace videostab;

DetectionPipeline::DetectionPipeline(char* detector, char* extractor, char* matcher) {
    this->detectorName = detector;
    this->extractorName = extractor;
    this->matcherName = matcher;
}



double sqr(double x) {
    return x * x;
}

bool fileExists(char* fileName) {
    struct stat buffer;
    return (stat(fileName, &buffer) == 0);
}

DetectionPipeline::~DetectionPipeline() {
}

size_t DetectionPipeline::runPipeline(char* image, Mat testImage, Mat testImageDescriptor, vector<KeyPoint> testImageKeypoints, bool showImages) {
    //Read image from file
    Mat im = imread(image, 1);
    
    
    
    vector<KeyPoint> imageKeypoints;
    
    Mat imageDescriptor;
    
    char storeName[1000];
    strcpy(storeName, image);
    strcat(storeName, "bin");
    
    if (fileExists(storeName)) {
    
    // Check if keypoints/descriptors are already stored
    FileStorage store(storeName, FileStorage::READ);
    
    FileNode keyNode = store["keypoints"];
    read(keyNode, imageKeypoints);
    FileNode descNode = store["descriptor"];
    read(descNode, imageDescriptor);
    
        store.release();
    } else {
    
    // Step 1: Detect Keypoints
    
    int minHessian = 400;
    //SiftFeatureDetector detector(minHessian);
    Ptr<FeatureDetector> detector = FeatureDetector::create(detectorName);
    detector->detect(im, imageKeypoints);
    
    // Step 2: Extract keypoints
    
    //SiftDescriptorExtractor extractor;
    
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(extractorName);
    extractor->compute(im, imageKeypoints, imageDescriptor);
    
        FileStorage store(storeName, FileStorage::WRITE);
        
        write(store, "keypoints", imageKeypoints);
        write(store, "descriptor", imageDescriptor);
        
        store.release();
        
    }
    // Step 3: Match Keypoints
    
    vector<vector<DMatch>> matchesImageToObject;
    vector<vector<DMatch>> matchesObjectToImage;
    
    vector<DMatch> matches;
    //matcher->match(imageDescriptor, testImageDescriptor, matchesImageToObject);
    //matcher->match(testImageDescriptor, imageDescriptor, matchesObjectToImage);
    
    int num_neighbors = 2;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    matcher->knnMatch(testImageDescriptor, imageDescriptor, matchesObjectToImage, num_neighbors);
    matcher->knnMatch(imageDescriptor, testImageDescriptor, matchesImageToObject, num_neighbors);
    
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
    
    
    Mat img_matches;
    /*
#ifdef DEBUG
    drawMatches( testImage, testImageKeypoints, im, imageKeypoints, matches, img_matches, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    imshow("Pre-Crosscheck Results", img_matches);
    
    waitKey(0);
#endif */
    //Remove top/bottom 20%
    
    float threshold = .2; //Top and Bottom threshold%
    int k = 0;
    for (int i = 0; i < matches.size(); i++) {
        Point2f p0 = testImageKeypoints.at(matches.at(i).queryIdx).pt;
        Point2f p1 = imageKeypoints.at(matches.at(i).trainIdx).pt;
        
        if (p0.y > (float)im.rows * threshold  &&
            p0.y < (float)im.rows*(1-threshold) &&
            p1.y > (float)testImage.rows * threshold   &&
            p1.y < (float)testImage.rows*(1-threshold))
        {
            matches[k] = matches[i];
            k++;
        }
    }
    matches.resize(k);/*
#ifdef DEBUG
    // Draw/Show matches
    drawMatches( testImage, testImageKeypoints, im, imageKeypoints, matches, img_matches, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    imshow("Crosscheck Results", img_matches);
    
    waitKey(0);
#endif*/
    
    // Find average x and y deltas
    
    float xDeltaAvg = 0;
    float yDeltaAvg = 0;
    
    for (DMatch m : matches) {
        Point2f p0 = testImageKeypoints.at(m.queryIdx).pt;
        Point2f p1 = imageKeypoints.at(m.trainIdx).pt;
        
        xDeltaAvg += p0.x - p1.x;
        
        yDeltaAvg += p0.y - p1.y;
    }
    xDeltaAvg = xDeltaAvg / matches.size();
    yDeltaAvg = yDeltaAvg / matches.size();
    
    // Eliminate matches that have too large of a y-delta
    float yDelta = 50;
    k = 0;
    for (int i = 0; i < matches.size(); i++) {
        Point2f p0 = testImageKeypoints.at(matches.at(i).queryIdx).pt;
        Point2f p1 = imageKeypoints.at(matches.at(i).trainIdx).pt;
        
        if (abs(p0.y - p1.y - yDeltaAvg) <= yDelta &&
            abs(p0.x - p1.x - xDeltaAvg) <= yDelta) {
            matches[k] = matches[i];
            k++;
        }
    }
    matches.resize(k);
    /*
#ifdef DEBUG
    // Draw/Show matches
    drawMatches( testImage, testImageKeypoints, im, imageKeypoints, matches, img_matches, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    imshow("Y-Delta Results", img_matches);
    
    waitKey(0);
#endif*/

    //imshow("Matches", img_matches);
    
    // Rigid
    if (matches.size() > 5) {
        vector<Point2f> obj;
        vector<Point2f> scene;
        
        for (int i = 0; i < matches.size(); i++) {
            obj.push_back(testImageKeypoints[matches[i].queryIdx].pt);
            scene.push_back(imageKeypoints[matches[i].queryIdx].pt);
        }
        
        Mat R = estimateRigidTransform(obj, scene, 0);
        
        if (R.rows != 0) {
        Mat_<float> r = R;
        
        int k = 0;
        float reprojectionError = 100;
        vector<DMatch> inlierMatches;
        for (DMatch match : matches) {
            Point2f p0 = testImageKeypoints.at(match.queryIdx).pt;
            Point2f p1 = imageKeypoints.at(match.trainIdx).pt;
            
            float x = r(0, 0) * p0.x + r(0, 1) * p0.y + r(0, 2);
            float y = r(1, 0) * p0.x + r(1, 1) * p0.y + r(1, 2);
            
            if (hypot(x - p1.x, y - p1.y) < reprojectionError) {
                inlierMatches.push_back(match);
            }
        }
        
        Mat img_matches;
        drawMatches( testImage, testImageKeypoints, im, imageKeypoints, inlierMatches, img_matches, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        imshow("Rigid", img_matches);
        
        waitKey(0);
            
            return inlierMatches.size();
        } else {
            return 0;
        }
    }
    
    // Affine
    if (matches.size() > 5 && false) {
        vector<Point2f> obj;
        vector<Point2f> scene;
    
        for (int i = 0; i < matches.size(); i++) {
            obj.push_back(testImageKeypoints[matches[i].queryIdx].pt);
            scene.push_back(imageKeypoints[matches[i].queryIdx].pt);
        }
        
        const RansacParams &params = RansacParams::affine2dMotionStd();
        vector<Point2f> best0, best1;
        float rmse = float(0.2);
        int numInliers = matches.size()/5;
        Mat H = estimateGlobalMotionRansac(obj, scene, 2, params, &rmse, &numInliers, 3, best0, best1);
        
        double ssd;
        
        vector<DMatch> trueMatches;
        vector<KeyPoint> objKey;
        vector<KeyPoint> imgKey;
        
        cout << "Size 0: " << best0.size() << " Size 2: " << best1.size() << endl;
        for (int i = 0; i < best0.size() && i < best1.size(); i++) {
            Point2f objPt = best0.at(i);
            Point2f scenePt = best1.at(i);
            
            double dist = sqrt(sqr(objPt.x - scenePt.x) + sqr(objPt.y - scenePt.y));
            
            trueMatches.push_back(DMatch(i, i, dist));
            objKey.push_back(KeyPoint(objPt, 1));
            imgKey.push_back(KeyPoint(scenePt, 1));
            ssd += dist;
        }
        
        //Visualize Matches
        drawMatches( testImage, objKey, im, imgKey, trueMatches, img_matches, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        //imshow("Affine", img_matches);
        
        waitKey(0);
        
        /*
        vector<Point3f> homPts;
        
        for (Point2f objPt : obj) {
            homPts.push_back(Point3f(objPt.x, objPt.y, 1));
        }
        
        vector<Point2f> actual;
        
        for (Point3f homPt : homPts) {
            Mat matPt = transform * Mat(homPt);
            Point2f pt = Point2f(matPt);
            actual.push_back(pt);
        }
        for (Point2f actualPt : actual) {
            for (Point2f scenePt : scene) {
                ssd += pow(sqrt(pow(actualPt.x - scenePt.x, 2) + pow(actualPt.y - scenePt.y, 2)), 2);
            }
        }*/
        
        return ssd;
    }
    
    // Homography
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


Mat DetectionPipeline::estimateGlobalMotionRansac(
                               InputArray points0, InputArray points1, int model, const cv::videostab::RansacParams &params,
                               float *rmse, int *ninliers, int iters, vector<Point2f> &best0, vector<Point2f> &best1)
{
    CV_Assert(model <= AFFINE);
    CV_Assert(points0.type() == points1.type());
    const int npoints = points0.getMat().checkVector(2);
    CV_Assert(points1.getMat().checkVector(2) == npoints);
    
    if (npoints < params.size)
        return Mat::eye(3, 3, CV_32F);
    
    const Point2f *points0_ = points0.getMat().ptr<Point2f>();
    const Point2f *points1_ = points1.getMat().ptr<Point2f>();
    const int niters = iters;
    
    // current hypothesis
    std::vector<int> indices(params.size);
    std::vector<Point2f> subset0(params.size);
    std::vector<Point2f> subset1(params.size);
    
    // best hypothesis
    std::vector<Point2f> subset0best(params.size);
    std::vector<Point2f> subset1best(params.size);
    Mat_<float> bestM;
    int ninliersMax = -1;
    
    RNG rng(0);
    Point2f p0, p1;
    float x, y;
    
    for (int iter = 0; iter < niters; ++iter)
    {
        for (int i = 0; i < params.size; ++i)
        {
            bool ok = false;
            while (!ok)
            {
                ok = true;
                indices[i] = static_cast<unsigned>(rng) % npoints;
                for (int j = 0; j < i; ++j)
                    if (indices[i] == indices[j])
                    { ok = false; break; }
            }
        }
        for (int i = 0; i < params.size; ++i)
        {
            subset0[i] = points0_[indices[i]];
            subset1[i] = points1_[indices[i]];
        }
        
        Mat_<float> M = cv::videostab::estimateGlobalMotionLeastSquares(subset0, subset1, model, 0);
        
        int numinliers = 0;
        for (int i = 0; i < npoints; ++i)
        {
            p0 = points0_[i];
            p1 = points1_[i];
            x = M(0,0)*p0.x + M(0,1)*p0.y + M(0,2);
            y = M(1,0)*p0.x + M(1,1)*p0.y + M(1,2);
            if (sqr(x - p1.x) + sqr(y - p1.y) < params.thresh * params.thresh)
                numinliers++;
        }
        if (numinliers >= ninliersMax)
        {
            bestM = M;
            ninliersMax = numinliers;
            subset0best.swap(subset0);
            subset1best.swap(subset1);
        }
    }
    
    if (ninliersMax < params.size)
        // compute RMSE
        bestM = cv::videostab::estimateGlobalMotionLeastSquares(subset0best, subset1best, model, rmse);
    else
    {
        subset0.resize(ninliersMax);
        subset1.resize(ninliersMax);
        for (int i = 0, j = 0; i < npoints && j < ninliersMax ; ++i)
        {
            p0 = points0_[i];
            p1 = points1_[i];
            x = bestM(0,0)*p0.x + bestM(0,1)*p0.y + bestM(0,2);
            y = bestM(1,0)*p0.x + bestM(1,1)*p0.y + bestM(1,2);
            if (sqr(x - p1.x) + sqr(y - p1.y) < params.thresh * params.thresh)
            {
                subset0[j] = p0;
                subset1[j] = p1;
                j++;
            }
        }
        bestM = cv::videostab::estimateGlobalMotionLeastSquares(subset0, subset1, model, rmse);
    }
    
    if (ninliers)
        *ninliers = ninliersMax;
    
    best0 = subset0best;
    best1 = subset1best;
    
    return bestM;
}


void DetectionPipeline::setObjType(char* type) {
    objType = type;
}

void DetectionPipeline::setImageName(char *name) {
    imageName = name;
}