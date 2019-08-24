//
// Created by tribbiani on 18-4-9.
//

#ifndef RGBD_SLAM_KEYPOINTEXTRACTION_H
#define RGBD_SLAM_KEYPOINTEXTRACTION_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>

#include "pca_analysis.h"
#include <string>

namespace keypoint
{
    struct keypointOption
    {
        float radius_featureCalculation;
        float ratioMax;
        float radiusNonMax;
        size_t minPtNum;

        keypointOption()
        {
            radius_featureCalculation = 0.4;
            radiusNonMax = 0.4;
            ratioMax = 0.92 ;
            minPtNum = 20;
        }
    };

    class keypointDetection: public pcaComponentAnalysis
    {
    public:
        keypointDetection(keypointOption keypointOption1)
        {
            kpO = keypointOption1;
        }

        bool keypointDetectionBasedonCurvature(const pcXYZptr &inputCloud, pcl::PointIndicesPtr &keypointIndices);



    private:
        bool pruneUnstablePoints(const std::vector<pcaFeature> &features, float ratioMax, pcl::PointIndicesPtr &indicesStable);

        bool nonMaxiamaSuppression(std::vector<pcaFeature> &features, pcl::PointIndicesPtr &indicesNonMaxSup);

        keypointOption kpO;
    };
}

#endif //RGBD_SLAM_KEYPOINTEXTRACTION_H
