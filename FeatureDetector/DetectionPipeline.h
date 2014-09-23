//
//  DetectionPipeline.h
//  FeatureDetector
//
//  Created by Akul Penugonda on 9/22/14.
//  Copyright (c) 2014 Akul Penugonda. All rights reserved.
//

#ifndef __FeatureDetector__DetectionPipeline__
#define __FeatureDetector__DetectionPipeline__

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

class DetectionPipeline {
public:
    void runPipeline(char* image);
    void setDir(char* dir);
    void setObjType(char* type);
    void setImageName(char* name);
private:
    char* detector;
    char* descriptor;
    char* matcher;
    char* directory;
    char* objType;
    char* objectOutput = "/Users/apenugonda/Pictures/OutputPics";
    char* imageName;
};


#endif /* defined(__FeatureDetector__DetectionPipeline__) */
