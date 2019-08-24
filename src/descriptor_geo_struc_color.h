//
// Created by tribbiani on 18-4-24.
//

#ifndef RGBD_SLAM_DESCRIPTOR_GEO_STRUC_COLOR_H
#define RGBD_SLAM_DESCRIPTOR_GEO_STRUC_COLOR_H

#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>

#include <vector>

#include "pca_analysis.h"

using namespace Eigen;

class descriptor_geometry_structure_color
{
public:
    struct neighborPt
    {
        double alpha , beta , gama;
        double a1d, a2d, a3d;
        double r, g, b;
        double curvature ;

        neighborPt()
        {
            alpha =0.0;
            beta =0.0;
            gama=0.0;

            a1d=0.0;
            a2d=0.0 ;
            a3d=0.0;

            r=0.0;
            g=0.0;
            b=0.0;

            curvature = 0.0;
        }
    };

    bool covarianceEstablish(vector<neighborPt> &neighborPts, MatrixXf &u, MatrixXf &covariance);

    bool averageDescriptor(vector<neighborPt> &neighborPts, MatrixXf &u);

    bool descriptor_GSC_extraction(const pcXYZRGBAptr &inputCloud, pcl::PointCloud<pcl::Normal> &normals,int keyptID, vector<neighborPt> &neighborPts);

};


#endif //RGBD_SLAM_DESCRIPTOR_GEO_STRUC_COLOR_H
