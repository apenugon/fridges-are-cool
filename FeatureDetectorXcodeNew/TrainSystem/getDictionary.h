/*
 *  getDictionary.h
 *  TrainSystem
 *
 *  Created by Vivek Krishnan on 10/15/14
 *  Copyright (c) 2014 Vivek Krishnan. All rights reserved.
 */

#ifndef __TrainSystem__MKSiftDictionary__
#define __TrainSystem__MKSiftDictionary__

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <stdio.h>
#include <string>

using namespace std;

void trainSupervisedLearner(string* trainFiles, unsigned n_files);



#endif
