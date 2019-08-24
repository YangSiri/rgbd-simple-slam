//
// Created by tribbiani on 18-3-24.
//
#include "vtkAutoInit.h"

#include <vector>
#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "DBoW3/DBoW3.h"

#include "slamBase.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    cout<<"Reading files ..."<<endl;
    vector<Mat> images;

    for(int i=1; i<1000 ; i++)
    {
        string path = "/home/tribbiani/workspace/rgbd_dataset_freiburg1_room_tum/rgb/"+to_string(i)+".png";
        images.push_back(imread(path));
    }

    //detect ORB features
    cout<<"dectecting features ..."<<endl;
    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("ORB");
    vector<cv::Mat> descriptor_all;

    for ( Mat& image:images )
    {
        vector<cv::KeyPoint> keypoints;
        cv::Ptr<cv::DescriptorExtractor> descriptor = cv::DescriptorExtractor::create("ORB");
        cv::Mat descriptors;

        detector->detect( image, keypoints );

//        float res_max = 0;
//        float res_min = 1;
//        for(int i=0 ; i<keypoints.size() ; i++)
//        {
//
//            if(keypoints[i].response > res_max )
//                res_max = keypoints[i].response;
//            if(keypoints[i].response < res_min )
//                res_min = keypoints[i].response;
//        }
//        for(int i=0 ; i<keypoints.size() ; i++)
//            if(keypoints[i].response < res_max )
//                keypoints.erase( keypoints.begin()+i );
//        cout<<"keypoints : "<<keypoints.size()<<endl;

        descriptor->compute(image,keypoints,descriptors);
        descriptor_all.push_back( descriptors.clone() );
    }
    //create vocabulary
    cout<<"Creating vocabulary ..."<<endl;
    DBoW3::Vocabulary vocab;
    vocab.create(descriptor_all);
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save("vocab_of_1000.yml.gz");

    cout<<"done"<<endl;

    return 0;
}