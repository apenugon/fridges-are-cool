//
//  ComparisonCheck.h
//  FeatureDetector
//
//  Created by Akul Penugonda on 10/9/14.
//  Copyright (c) 2014 Akul Penugonda. All rights reserved.
//

#ifndef __FeatureDetector__ComparisonCheck__
#define __FeatureDetector__ComparisonCheck__

#include "DetectionPipeline.h"
#include <string>

class ComparisonCheck {
public:
    ComparisonCheck(char* detector, char* extractor, char* matcher);
    ~ComparisonCheck();
    int runCheck(char* imagePath, char* actual);
    static bool has_suffix(const std::string &str, const std::string &suffix);
    static char* subdirString(char* parent, char* name);
private:
    DetectionPipeline *pipeline;
    char* detector;
    char* extractor;
    char* matcher;
};



#endif /* defined(__FeatureDetector__ComparisonCheck__) */
