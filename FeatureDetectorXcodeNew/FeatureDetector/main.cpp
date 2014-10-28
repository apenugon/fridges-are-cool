//
//  main.cpp
//  FeatureDetector
//
//  Created by Akul Penugonda on 9/22/14.
//  Copyright (c) 2014 Akul Penugonda. All rights reserved.
//

#include "localdefs.h"
#include "DetectionPipeline.h"
#include "ComparisonCheck.h"
#include <stdio.h>
#include <iostream>
#include <dirent.h>

using namespace std;

//#define TESTING_DATA "/Users/apenugonda/Pictures/Objects-Test-640"

int fileIterate(char* detector, char* extractor, char* matcher) {
    int total = 0;
    int correct = 0;
    //Iterate through subdirectories
    DIR *d;
    struct dirent *dir;
    d = opendir(TESTING_DATA);
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
            subdirName = ComparisonCheck::subdirString(TESTING_DATA, dir->d_name);
            DIR *d2;
            struct dirent *dir2;
            d2 = opendir(subdirName);
            while ((dir2 = readdir(d2)) != NULL) {
                
                //Can only be a JPG file
                if (!ComparisonCheck::has_suffix(dir2->d_name, ".jpg")) {
                    continue;
                }
                
                imageName = ComparisonCheck::subdirString(subdirName, dir2->d_name);
                total++;
                ComparisonCheck *newCheck = new ComparisonCheck(detector, extractor, matcher);
                if (newCheck->runCheck(imageName, dir->d_name))
                    correct++;
                
                delete newCheck;
                free(imageName);
            }
            free(subdirName);
        }
    }
    printf("Detector: %s, Extractor: %s, Matcher: %s Correct: %i", detector, extractor, matcher, correct);
    return correct;
}

int handleError( int status, const char* func_name,
                const char* err_msg, const char* file_name,
                int line, void* userdata )
{
    //Do nothing -- will suppress console output
    return 0;   //Return value is not used
}

int main(int argc, const char * argv[])
{
    //Initialize nonfree module
    cv::initModule_nonfree();

    vector<char*> detectors = {"FAST", "STAR", "SIFT", "SURF", "ORB", "BRISK", "MSER", "GFTT", "HARRIS"};
    vector<char*> extractors = {"SIFT", "SURF", "BRIEF", "BRISK", "ORB", "FREAK"};
    vector<char*> matchers = {"BruteForce-Hamming", "BruteForce", "BruteForce-L1",  "BruteForce-Hamming(2)", "FlannBased"};
    
    vector<tuple<char*, char*>> testFiles;
    /*
    //For each file, iterate through every type. Print out the number correct
    cv::redirectError(handleError);
    int max = 0;
    char* bestDetector = "None";
    char* bestExtractor = "None";
    char* bestMatcher = "None";
    for (char* detector : detectors) {
        for (char* extractor : extractors) {
            for (char* matcher : matchers) {
                try {
                ComparisonCheck *newCheck = new ComparisonCheck(detector, extractor, matcher);
                int correct = newCheck->runCheck("/Users/apenugonda/Pictures/Objects-Test-640/whippedcream/test-16.jpg", "whippedcream");
                
                printf("Detector: %s, Extractor: %s, Matcher: %s, Correct: %i\n", detector, extractor, matcher, correct);
                /*int correct = fileIterate(detector, extractor, matcher);
                if (correct > max) {
                    bestDetector = detector;
                    bestExtractor = extractor;
                    bestMatcher = matcher;
                    max = correct;
                }
                } catch(cv::Exception& e) {
                    printf("Not allowed: Detector: %s, Extractor: %s, Matcher: %s", detector, extractor, matcher);
                }
            }
        }
    }*/
    //printf("Best Stats: Detector: %s, Extractor: %s, Matcher: %s, Correct: %i", bestDetector, bestExtractor, bestMatcher, max);
    ComparisonCheck *newCheck = new ComparisonCheck("SIFT", "SIFT", "BruteForce");
    newCheck->runCheck("/Users/apenugonda/Pictures/Objects-Test-640/cactus/test-11.jpg", "cactus");
    delete newCheck;
}

