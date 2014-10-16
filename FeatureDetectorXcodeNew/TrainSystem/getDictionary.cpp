#include "trainConstants.h"
#include "getDictionary.h"

#include <dirent.h>
#include <sys/types.h>

using namespace std;
using namespace cv;

void getDictionary(string* trainFiles, unsigned n_files)
{
    printf("Computing Dictionary of Visual Words\n");
    //Training Set Variables
    Mat feature_responses;              //store the features from all of the images
    Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");   //The sift feature extractor
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");

    //Single Image Variables
    Mat im;                         //Store the current image
    Mat descriptor;
    vector<KeyPoint> keypoints;     //Store the keypoints of the current image

    for (int i = 0; i < n_files; i++)
    {
        //Get the filename of the next image
        string filename = trainFiles[i];
        //Read the image in
        im = imread(filename.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        //Generate Keypoints
        detector->detect(im, keypoints);
        extractor->compute(im, keypoints, descriptor);

        //Update the feature_responses
        feature_responses.push_back(descriptor);
    }

    //Set up the KMeans Cluster Trainer
    TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
    int retries = 1;
    int flags = KMEANS_PP_CENTERS;
    BOWKMeansTrainer bowTrainer(dictionary_size(), tc, retries, flags);

    //Compute the dictionary using kmeans
    Mat dictionary = bowTrainer.cluster(feature_responses);

    cv::FileStorage storage("dictionary.yml", cv::FileStorage::WRITE);
    storage << "dictionary" << dictionary;
    storage.release();
}

void trainSystem(string* train_files, int n_files)
{
    cv::FileStorage r_storage("dictionary.yml", cv::FileStorage::READ);
    Mat dictionary;
    r_storage["dictionary"] >> dictionary;
    r_storage.release();

    Mat im;

    Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");   //The sift feature extractor
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

    BOWImgDescriptorExtractor dextract(extractor, matcher);
    dextract.setVocabulary(dictionary);

    vector<KeyPoint> keypoints;
    Mat descriptor;
    Mat train_data;

    for (int i = 1; i < n_files; ++i)
    {
        im = imread(train_files[i].c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        detector->detect(im, keypoints);
        dextract.compute(im, keypoints, descriptor);
        train_data.push_back(descriptor);
    }

    cv::FileStorage storage("trainOutput.yml", cv::FileStorage::WRITE);
    storage << "trainHists" << train_data;
    storage.release();
}

void classifyImage(Mat dictionary, BOWImgDescriptorExtractor bowDE, char *im_filename)
{
    Mat im = imread(im_filename);
    SiftDescriptorExtractor detector;   //The sift feature extractor

    vector<KeyPoint> keypoints;
    detector.detect(im, keypoints);

    Mat descriptor;
    bowDE.compute(im, keypoints, descriptor);
}



int main()
{
    DIR *dir;
    struct dirent *ent;
    vector<string> files;
    if ((dir = opendir("./images/")) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            char buf[100];
            sprintf(buf, "./images/%s", ent->d_name);
            files.push_back(string(buf));
        }
        closedir(dir);
    } else {
        perror("");
        return EXIT_FAILURE;
    }

    getDictionary(&files[0], files.size());
    trainSystem(&files[0], files.size());
    printf("Computed Dictionary\n");

    return 0;
}
