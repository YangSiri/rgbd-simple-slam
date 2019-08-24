//
// Created by tribbiani on 18-4-9.
//

#include "pca_analysis.h"

#include <vector>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <opencv2/core/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>

#include <tbb/concurrent_vector.h>
#include <tbb/compat/ppl.h>
#include <ml.h>

using namespace std;
using namespace pcl;

bool pcaComponentAnalysis::calculatePCAfeaturesOfPointCloud(const pcXYZptr &inputCloud, float radius, std::vector<pcaFeature> &features)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdTreeFLANN;
    kdTreeFLANN.setInputCloud(inputCloud);


    Concurrency::parallel_for(size_t(0),inputCloud->points.size(), [&](size_t i)
    {
        vector<int> searchIndices;
        vector<float > distances;
        vector<int>().swap(searchIndices);
        vector<float>().swap(distances);

        if(!isfinite(inputCloud->points[i].x)||!isfinite(inputCloud->points[i].y))
            return 0;

        kdTreeFLANN.radiusSearch(i,radius,searchIndices,distances);
        features[i].pt = inputCloud->points[i];
        features[i].ptnum = searchIndices.size();
        features[i].ptId = i;

        calculatePCAfeature(inputCloud, searchIndices, features[i]);
    });

    return true;
}

bool pcaComponentAnalysis::calculatePCAfeature(const pcXYZptr &inputCloud, const std::vector<int> &searchIndices, pcaFeature &feature)
{
    size_t ptNum;
    ptNum = searchIndices.size();

    if(ptNum<3)
        return false;
    CvMat* pData = cvCreateMat(ptNum,3,CV_32FC1);
    CvMat* pMean = cvCreateMat(1,3,CV_32FC1);
    CvMat* pEigVals = cvCreateMat(1,3,CV_32FC1);
    CvMat* pEigVecs = cvCreateMat(3,3,CV_32FC1);

    for(size_t i =0; i<ptNum ; ++i)
    {
        cvmSet(pData, i, 0, inputCloud->points[searchIndices[i]].x);
        cvmSet(pData, i, 1, inputCloud->points[searchIndices[i]].y);
        cvmSet(pData, i, 2, inputCloud->points[searchIndices[i]].z);
    }

    cvCalcPCA(pData, pMean, pEigVals, pEigVecs, CV_PCA_DATA_AS_ROW);

    feature.vectors.principleDirect[0] = cvmGet(pEigVecs, 0 ,0);
    feature.vectors.principleDirect[1] = cvmGet(pEigVecs, 0 ,1);
    feature.vectors.principleDirect[2] = cvmGet(pEigVecs, 0 ,2);

    feature.vectors.middleDirect[0] = cvmGet(pEigVecs, 1 ,0);
    feature.vectors.middleDirect[1] = cvmGet(pEigVecs, 1 ,1);
    feature.vectors.middleDirect[2] = cvmGet(pEigVecs, 1 ,2);

    feature.vectors.normalDirect[0] = cvmGet(pEigVecs, 2 ,0);
    feature.vectors.normalDirect[1] = cvmGet(pEigVecs, 2 ,1);
    feature.vectors.normalDirect[2] = cvmGet(pEigVecs, 2 ,2);

    feature.values.lamda1 = cvmGet(pEigVals,0,0);
    feature.values.lamda2 = cvmGet(pEigVals,0,1);
    feature.values.lamda3 = cvmGet(pEigVals,0,2);

    if ((feature.values.lamda1 + feature.values.lamda2 + feature.values.lamda3) == 0)
    {
        feature.curvature = 0;
    }
    else
    {
        feature.curvature = feature.values.lamda3 / (feature.values.lamda1 + feature.values.lamda2 + feature.values.lamda3);

    }

    cvReleaseMat(&pEigVecs);
    cvReleaseMat(&pEigVals);
    cvReleaseMat(&pMean);
    cvReleaseMat(&pData);
    return true;

}

/*
bool keypointNeiborAnalysis::calculateNeiborPoints(const pcXYZptr &originalCloud,
                                                   const pcl::PointIndicesPtr &keypointIndices,
                                                   vector<localReferFrame> &localReferFrames,double neiborRadius)
{
//    vector<neiborPoint> neiborPts;
//    vector<neiborPoint>().swap(neiborPts);
    ///设置搜索半径
//    float neiborRadius = 0.5 ;
    pcl::KdTreeFLANN<PointXYZ> kdTreeFLANN;
    kdTreeFLANN.setInputCloud(originalCloud);


//    Concurrency::parallel_for(size_t(0),keypointIndices->indices.size(), [&](size_t i)
    for(size_t i=0 ; i<keypointIndices->indices.size() ; ++i)
    {
        vector<int> searchIndices;
        vector<float> distance;
        vector<int> neiborSearch;
        vector<float> neiborSearchDist;

        vector<int>().swap(searchIndices);
        vector<float>().swap(distance);

        kdTreeFLANN.radiusSearch(*originalCloud,keypointIndices->indices[i],neiborRadius,searchIndices,distance);
//        cout<<"  "<<searchIndices.size() <<endl;
        neiborPoint neiborPoint1;//template
        neiborPoint1.keypointID =keypointIndices->indices[i];
        neiborPoint1.keypt = originalCloud->points[keypointIndices->indices[i]];
        neiborPoint1.ptID = neiborSearch ;
        neiborPoint1.totalWeight = 0.0;

        ///遍历每个邻域点
        for(size_t j=0 ; j<searchIndices.size() ; ++j)
        {
            vector<int>().swap(neiborSearch);
            vector<float>().swap(neiborSearchDist);

            neiborPoint1.pts.push_back(originalCloud->points[searchIndices[j]]) ;

            neiborPoint1.wDistance.push_back( (neiborRadius - distance[j]) / neiborRadius );
            kdTreeFLANN.radiusSearch(*originalCloud,searchIndices[j],neiborRadius * 0.5,neiborSearch,neiborSearchDist);
            neiborPoint1.wDensity.push_back( 1.0 / (neiborSearch.size()+1) ) ;

            neiborPoint1.totalWeight += neiborPoint1.wDistance[j] * neiborPoint1.wDensity[j];
        }

        localReferFrame localReferFrame1;
        solveLocalreferFrame(neiborPoint1,localReferFrame1);
        localReferFrames.push_back(localReferFrame1);
//        neiborPt.push_back(neiborPoint1);


    }

    return true;

}
*/

bool keypointNeiborAnalysis::solveLocalreferFrame(neiborPoint &neiborPt,
                                                  localReferFrame &lcf)
{
    cv::Mat covarianceM(3,3,CV_32FC1);

    ///计算每个关键点处的协方差矩阵M
    for(size_t j=0 ; j<neiborPt.pts.size() ; ++j)
    {
        covarianceM.at<float>(0,0) += ( neiborPt.wDensity[j] *  neiborPt.wDistance[j]
                                        * (neiborPt.pts[j].x - neiborPt.keypt.x)*(neiborPt.pts[j].x - neiborPt.keypt.x) )/neiborPt.totalWeight;

        covarianceM.at<float>(0,1) += ( neiborPt.wDensity[j] *  neiborPt.wDistance[j]
                                        * (neiborPt.pts[j].x - neiborPt.keypt.x)*(neiborPt.pts[j].y - neiborPt.keypt.y) )/neiborPt.totalWeight;

        covarianceM.at<float>(0,2) += ( neiborPt.wDensity[j] *  neiborPt.wDistance[j]
                                        * (neiborPt.pts[j].x - neiborPt.keypt.x)*(neiborPt.pts[j].z - neiborPt.keypt.z) )/neiborPt.totalWeight;

        covarianceM.at<float>(1,0) = covarianceM.at<float>(0,1);

        covarianceM.at<float>(1,1) += ( neiborPt.wDensity[j] *  neiborPt.wDistance[j]
                                        * (neiborPt.pts[j].y - neiborPt.keypt.y)*(neiborPt.pts[j].y - neiborPt.keypt.y) )/neiborPt.totalWeight;

        covarianceM.at<float>(1,2) += ( neiborPt.wDensity[j] *  neiborPt.wDistance[j]
                                        * (neiborPt.pts[j].y - neiborPt.keypt.y)*(neiborPt.pts[j].z - neiborPt.keypt.z) )/neiborPt.totalWeight;

        covarianceM.at<float>(2,0) = covarianceM.at<float>(0,2);

        covarianceM.at<float>(2,1) = covarianceM.at<float>(1,2);

        covarianceM.at<float>(2,2) += ( neiborPt.wDensity[j] *  neiborPt.wDistance[j]
                                        * (neiborPt.pts[j].z - neiborPt.keypt.z)*(neiborPt.pts[j].z - neiborPt.keypt.z) )/neiborPt.totalWeight;
    }
    //covarianceMs.push_back(covarianceM);
    ///计算M的特征向量和特征值
    cv::eigen(covarianceM,lcf.lamda, lcf.eVec);
    lcf.keypoint = neiborPt.keypt;

    ///建立以关键点为中心，特征向量e1为X轴，e2为Y轴的局部坐标系并计算与原坐标系的转换关系
    pcl::PointCloud<pcl::PointXYZ> cloud_src;
    pcl::PointCloud<pcl::PointXYZ> cloud_target;

    pcl::PointXYZ pt;

    pt.x = 1;
    pt.y = 0;
    pt.z = 0;
    cloud_src.push_back(pt);

    pt.x = 0;
    pt.y = 1;
    pt.z = 0;
    cloud_src.push_back(pt);

    pt.x = 0;
    pt.y = 0;
    pt.z = 1;
    cloud_src.push_back(pt);

    pt.x = 0;
    pt.y = 0;
    pt.z = 0;
    cloud_src.push_back(pt);

    pt.x = lcf.keypoint.x + lcf.eVec.at<float>(0,0);
    pt.y = lcf.keypoint.y + lcf.eVec.at<float>(0,1);
    pt.z = lcf.keypoint.z + lcf.eVec.at<float>(0,2);
    cloud_target.push_back(pt);

    pt.x = lcf.keypoint.x + lcf.eVec.at<float>(1,0);
    pt.y = lcf.keypoint.y + lcf.eVec.at<float>(1,1);
    pt.z = lcf.keypoint.z + lcf.eVec.at<float>(1,2);
    cloud_target.push_back(pt);

    pt.x = lcf.keypoint.x + ( lcf.eVec.at<float>(0,1)* lcf.eVec.at<float>(1,2) - lcf.eVec.at<float>(1,1)*lcf.eVec.at<float>(0,2) );
    pt.y = lcf.keypoint.y + ( lcf.eVec.at<float>(0,2)* lcf.eVec.at<float>(1,0) - lcf.eVec.at<float>(1,2)*lcf.eVec.at<float>(0,0) );
    pt.z = lcf.keypoint.z + ( lcf.eVec.at<float>(0,0)* lcf.eVec.at<float>(1,1) - lcf.eVec.at<float>(0,1)*lcf.eVec.at<float>(1,0) );
    cloud_target.push_back(pt);

    cloud_target.push_back(lcf.keypoint);

    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>  transformationEstimation;
    transformationEstimation.estimateRigidTransformation(cloud_src,cloud_target,lcf.trans);

    return true;

}

bool sphereCalcu(keypointNeiborAnalysis::neiborPoint &neiborPt, keypointNeiborAnalysis::localReferFrame &lcf, vector<float> &distances, double radius ,int num)
{
    lcf.score.resize(num+1);
    double max_distance = 0.0;
//    for(int i=0 ; i<neiborPt.pts.size() ; i++)
//    {
//        if(distances[i] > max_distance)
//            max_distance = distances[i];
//    }
//    cout<<"Max distance : "<<max_distance<<endl;
    for(int i=0 ; i<neiborPt.pts.size() ; i++)
    {
        for(int j=1 ; j<=num ; ++j)
            if(distances[i] < (radius / num)*j)
                lcf.score[j] +=( neiborPt.wDistance[i] * neiborPt.wDistance[i] ) / neiborPt.totalWeight;

    }
}

bool keypointNeiborAnalysis::calculateNeiborPoints(const pcXYZptr &originalCloud,
                                                   const pcl::PointIndicesPtr &keypointIndices,
                                                   vector<localReferFrame> &localReferFrames,
                                                   double neiborRadius, int num)
{
//    vector<neiborPoint> neiborPts;
//    vector<neiborPoint>().swap(neiborPts);
    ///设置搜索半径
//    float neiborRadius = 0.5 ;
    pcl::KdTreeFLANN<PointXYZ> kdTreeFLANN;
    kdTreeFLANN.setInputCloud(originalCloud);


//    Concurrency::parallel_for(size_t(0),keypointIndices->indices.size(), [&](size_t i)
    for(size_t i=0 ; i<keypointIndices->indices.size() ; ++i)
    {
        vector<int> searchIndices;
        vector<float> distance;
        vector<int> neiborSearch;
        vector<float> neiborSearchDist;

        vector<int>().swap(searchIndices);
        vector<float>().swap(distance);

        kdTreeFLANN.radiusSearch(*originalCloud,keypointIndices->indices[i],neiborRadius,searchIndices,distance);
//        cout<<"neighborPoints : "<<searchIndices.size() <<endl;
        neiborPoint neiborPoint1;//template
        neiborPoint1.keypointID =keypointIndices->indices[i];
        neiborPoint1.keypt = originalCloud->points[keypointIndices->indices[i]];
//        neiborPoint1.ptID = neiborSearch ;
        neiborPoint1.totalWeight = 0.0;

        ///遍历每个邻域点
        for(size_t j=0 ; j<searchIndices.size() ; ++j)
        {
            vector<int>().swap(neiborSearch);
            vector<float>().swap(neiborSearchDist);

            neiborPoint1.pts.push_back(originalCloud->points[searchIndices[j]]) ;

            neiborPoint1.wDistance.push_back( (neiborRadius - distance[j]) / neiborRadius );
            kdTreeFLANN.radiusSearch(*originalCloud,searchIndices[j],neiborRadius * 0.5,neiborSearch,neiborSearchDist);
            neiborPoint1.wDensity.push_back( 1.0 / (neiborSearch.size()+1) ) ;

            neiborPoint1.totalWeight += neiborPoint1.wDistance[j] * neiborPoint1.wDensity[j];
        }
        localReferFrame lcf;
        sphereCalcu(neiborPoint1,lcf, distance, neiborRadius, num);
        localReferFrames.push_back(lcf);
//        neiborPt.push_back(neiborPoint1);
    }


    return true;
}

