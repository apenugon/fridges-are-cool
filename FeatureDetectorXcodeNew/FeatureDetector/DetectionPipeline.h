//
//  DetectionPipeline.h
//  FeatureDetector
//
//  Created by Akul Penugonda on 9/22/14.
//  Copyright (c) 2014 Akul Penugonda. All rights reserved.
//

#ifndef __FeatureDetector__DetectionPipeline__
#define __FeatureDetector__DetectionPipeline__

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <stdio.h>

class DetectionPipeline {
public:
    DetectionPipeline(char* detector, char* extractor, char* matcher);
    ~DetectionPipeline();
    size_t runPipeline(char* image, cv::Mat testImage, cv::Mat testImageDescriptor, std::vector<cv::KeyPoint> testImageKeypoints, bool showImages);
    void setDir(char* dir);
    void setObjType(char* type);
    void setImageName(char* name);
private:
    char* detectorName;
    char* extractorName;
    char* matcherName;
    char* directory;
    char* objType;
    char* objectOutput = "/Users/apenugonda/Pictures/OutputHomography";
    char* imageName;
};


#endif /* defined(__FeatureDetector__DetectionPipeline__) */
