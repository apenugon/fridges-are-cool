//
//  main.cpp
//  FeatureDetector
//
//  Created by Akul Penugonda on 9/22/14.
//  Copyright (c) 2014 Akul Penugonda. All rights reserved.
//

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, const char * argv[])
{
    //Put whatever image you want here - I'm using the Kefir
    Mat im = imread(argc == 2 ? argv[1] : "/Users/apenugonda/Documents/FeatureDetect/FeatureDetector/keffront.jpg", 1);
    if (im.empty()) {
        cout << "Cannot open image!" << endl;
        return -1;
    }
    
    imshow("image", im);
    
    //Perform Keypoint Finding
    
    int minHessian = 400;
    SiftFeatureDetector detector(minHessian);
    
    vector<KeyPoint> imageKeypoints;
    
    detector.detect(im, imageKeypoints);
    
    //Draw Keypoints on Image
    
    Mat keypointsImage;
    
    drawKeypoints(im, imageKeypoints, keypointsImage, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    imshow("Keypoints", keypointsImage);
    
    //Write image to file
    imwrite("/Users/apenugonda/Pictures/ItemPics/output.jpg", keypointsImage);
    
    waitKey(0);
    
    return 0;
}

