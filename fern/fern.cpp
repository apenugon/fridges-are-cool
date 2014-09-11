#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/legacy/legacy.hpp"

#include <algorithm>
#include <iostream>
#include <vector>
#include <stdio.h>

//Should work on all POSIX compliant systems
#include <dirent.h>

#define TRAININGDATA "./images/train"

using namespace cv;
int main(int argc, char** argv)
{
	//load scene (first arg)
	const char* scene_filename = argc > 1 ? argv[1] : "box_in_scene.png";
	Mat scene = imread(scene_filename, CV_LOAD_IMAGE_GRAYSCALE);

	//quit if cannot find query
	if (!scene.data)
	{
		fprintf(stderr, "Cannot load %s\n", scene_filename);
		exit(-1);
	}

    //Generic initializations and for scene
	Mat image;
	double imgscale = 1;
	resize(scene,image,Size(), 1./imgscale, 1./imgscale, INTER_CUBIC);
	Size patchSize(32, 32);
	LDetector ldetector(7, 20, 2, 2000, patchSize.width, 2);
	ldetector.setVerbose(true);
    vector<Mat> imgpyr;
    int blurKSize = 3;
    double sigma = 0;
    GaussianBlur(image, image, Size(blurKSize, blurKSize), sigma, sigma);
    buildPyramid(image, imgpyr, ldetector.nOctaves-1);
    vector<KeyPoint> imgKeypoints;
    PatchGenerator gen(0,256,5,true,0.8,1.2,-CV_PI/2,CV_PI/2,-CV_PI/2,CV_PI/2);

    //Traverse directories
	DIR *d;
	struct dirent *dir;
	d = opendir(TRAININGDATA);
	if (d)
	{
		//for each directory located here (really all files, but should only be
		//directories here)
		while ((dir = readdir(d)) != NULL)
		{
			//Don't count the current directory or parent directory
			if (!strcmp(dir->d_name,".") || !strcmp(dir->d_name,".."))
				continue;
			printf("Loading object: %s\n", dir->d_name);
			DIR *d2;
			struct dirent *dir2;

			//Figure out filename (I sure hope this works)
			char *new_str = (char *)malloc(strlen(dir->d_name)+strlen(TRAININGDATA)+2);
			new_str[0]='\0';
			strcat(new_str,TRAININGDATA);
			strcat(new_str,"/");
			strcat(new_str,dir->d_name);

			d2 = opendir(new_str);
			if (d2)
			{
				//for each image that is a picture of the object
				while ((dir2 = readdir(d2)) != NULL)
				{
					//Don't count the current directory or parent directory
					if (!strcmp(dir2->d_name,".") || !strcmp(dir2->d_name,".."))
						continue;
                    //Also do not pick files ending in .gz
                    char *dot = strrchr(dir2->d_name,'.');
                    if (dot && !strcmp(dot,".gz"))
                        continue;

					printf("Loading image: %s\n", dir2->d_name);

					//get full filename of the image we want to load
					char *new_str2 = (char *)malloc(strlen(new_str)+strlen(dir2->d_name)+2);
					new_str2[0]='\0';
					strcat(new_str2,new_str);
					strcat(new_str2,"/");
					strcat(new_str2,dir2->d_name);

					//try actually loading picture
					Mat objectpic = imread(new_str2,CV_LOAD_IMAGE_GRAYSCALE);
					if (!objectpic.data)
					{
						fprintf(stderr, "WARNING: Could not load data from %s\n",new_str2);
                        exit(-1);
					}
	                PlanarObjectDetector detector;
                    vector<Mat> objpyr;
                    GaussianBlur(objectpic, objectpic, Size(blurKSize, blurKSize), sigma, sigma);
                    buildPyramid(objectpic, objpyr, ldetector.nOctaves-1);
                    vector<KeyPoint> objKeypoints;

                    //Model Filename for recovering and such
                    string model_filename = format("%s_model.xml.gz", new_str2);
                    printf("Trying to load %s ...\n", model_filename.c_str());
                    FileStorage fs(model_filename, FileStorage::READ);
                    if ( fs.isOpened() )
                    {
                        //does this mean we need a new detector for each one?
                        detector.read(fs.getFirstTopLevelNode());
                        printf("Successfully loaded %s.\n", model_filename.c_str());
                    }
                    else
                    {
                        printf("The file not found and can not be read. Let's train the model.\n");
                        printf("Step 1. Finding the robust keypoints ...\n");
                        ldetector.setVerbose(true);
                        ldetector.getMostStable2D(objectpic, objKeypoints, 100, gen);
                        printf("Done.\nStep 2. Training ferns-based planar object detector ...\n");
                        detector.setVerbose(true);

                        detector.train(objpyr, objKeypoints, patchSize.width, 100, 11, 10000, ldetector, gen);
                        printf("Done.\nStep 3. Saving the model to %s ...\n", model_filename.c_str());
                        if( fs.open(model_filename, FileStorage::WRITE) )
                            detector.write(fs, "ferns_model");
                    }
                    //printf("Now find the keypoints in the image, try recognize them and compute the homography matrix\n");
                    fs.release();

                    vector<int> pairs;
                    Mat H;
                    vector<Point2f> dst_corners;

                    double t = (double)getTickCount();
                    objKeypoints = detector.getModelPoints();
                    ldetector(imgpyr, imgKeypoints, 300);

                    //std::cout << "Object keypoints: " << objKeypoints.size() << "\n";
                    //std::cout << "Image keypoints: " << imgKeypoints.size() << "\n";
                    bool found = detector(imgpyr, imgKeypoints, H, dst_corners, &pairs);
                    t = (double)getTickCount() - t;
                    //printf("%gms\n", t*1000/getTickFrequency());

                    printf("Number of matches: %ld\n", pairs.size());
                    //TODO: Keep track of top match so far
				}

				//we are done with the directory
				closedir(d2);
			}
			else
			{
				fprintf(stderr, "Could not open local file.\n");
			}
		}

		//we are done with the directory
		closedir(d);
	}
	else
	{
		fprintf(stderr, "Could not open local directory.\n");
	}


	return 0;
}
