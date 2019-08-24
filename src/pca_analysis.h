//
// Created by tribbiani on 18-4-9.
//

#ifndef RGBD_SLAM_PCA_ANALYSIS_H
#define RGBD_SLAM_PCA_ANALYSIS_H

#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>

#include <vector>
using namespace std;
typedef pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcXYZRGBAptr;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr pcXYZptr;
typedef pcl::PointCloud<pcl::PointNormal>::Ptr pcPointNormalptr;
typedef pcl::PointCloud<pcl::Normal>::Ptr pcNormalptr;

class pcaComponentAnalysis
{
public:
    struct eigenValues {
        double lamda1;
        double lamda2;
        double lamda3;
    };

    struct eigenVectors {
        //3X1 vector
        Eigen::Vector3f principleDirect;
        Eigen::Vector3f middleDirect;
        Eigen::Vector3f normalDirect;
    };

    struct pcaFeature {
        eigenValues values;
        eigenVectors vectors;
        size_t ptnum;
        size_t ptId;
        pcl::PointXYZ pt;
        double curvature;

    };

    bool calculatePCAfeaturesOfPointCloud(const pcXYZptr &inputCloud, float radius, std::vector<pcaFeature> &features);

    bool calculatePCAfeature(const pcXYZptr &inputCloud, const std::vector<int> &searchIndices, pcaFeature &feature);


};

class keypointNeiborAnalysis
{
public:

    struct localReferFrame {

        cv::Mat lamda;
        cv::Mat eVec;
        pcl::PointXYZ keypoint;
        Eigen::Matrix4f trans;
        vector<float> score;

    };

    struct neiborPoint {

        int keypointID;
        pcl::PointXYZ keypt;
//        vector<int> ptID;
//        vector<float> score;
        vector<float> wDensity;
        vector<float> wDistance;
        pcl::PointCloud<pcl::PointXYZ> pts;
        float totalWeight;//sum(wDensity * wDistance)
    };

    bool calculateNeiborPoints(const pcXYZptr &originalCloud, const pcl::PointIndicesPtr &keypointIndices,
                               vector<localReferFrame> &localReferFrames,
                               double neiborRadius, int num);

    bool solveLocalreferFrame(neiborPoint &neiborPoints,localReferFrame  &localReferFrames );


};

#endif //RGBD_SLAM_PCA_ANALYSIS_H