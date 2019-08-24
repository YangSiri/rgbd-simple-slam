//
// Created by tribbiani on 18-4-24.
//
#include "pca_analysis.h"
#include "descriptor_geo_struc_color.h"

#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <math.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

bool lamda_ordered(double &a, double &b, double &c)
{
    double t;
    if(a<b)
    {
        t=a;
        a=b;
        b=t;
    }

    if(a<c)
    {
        t=a;
        a=c;
        c=t;
    }

    if(c>b)
    {
        t=b;
        b=c;
        c=t;
    }
}

double angle_of_vector(Vector3f a, Vector3f b )
{
    double normA = sqrt(a(0,0)*a(0,0)+a(1,0)*a(1,0)+a(2,0)*a(2,0));
    double normB = sqrt(b(0,0)*b(0,0)+b(1,0)*b(1,0)+b(2,0)*b(2,0));
    double cos= (a(0,0)*b(0,0) + a(1,0)*b(1,0) + a(2,0)*b(2,0)) / (normA * normB);
    double angle = acos(cos) * 180 / M_PI;
    return  angle;

}

bool pcaFeature(const pcXYZRGBAptr &inputCloud, pcaComponentAnalysis::pcaFeature &feature, vector<int> searchIndices)
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

bool descriptor_geometry_structure_color::averageDescriptor(vector<neighborPt> &neighborPts, MatrixXf &u)
{
    for(int i=0 ; i<neighborPts.size() ; ++i)
    {
        u(0,0) += neighborPts[i].alpha / neighborPts.size();
        u(0,1) += neighborPts[i].beta / neighborPts.size();
        u(0,2) += neighborPts[i].gama / neighborPts.size();

        u(0,3) += neighborPts[i].a1d / neighborPts.size();
        u(0,4) += neighborPts[i].a2d / neighborPts.size();
        u(0,5) += neighborPts[i].a3d / neighborPts.size();

        u(0,6) += neighborPts[i].r / neighborPts.size();
        u(0,7) += neighborPts[i].g / neighborPts.size();
        u(0,8) += neighborPts[i].b / neighborPts.size();

        u(0,9) += neighborPts[i].curvature / neighborPts.size();
    }

}

bool descriptor_geometry_structure_color::covarianceEstablish(vector<neighborPt> &neighborPts, MatrixXf &u, MatrixXf &covariance)
{
    int num = neighborPts.size();
    for(int i=0 ; i<num ; ++i)
    {
        MatrixXf f(1,10);
        f(0,0) = neighborPts[i].alpha;
        f(0,1) = neighborPts[i].beta;
        f(0,2) = neighborPts[i].gama;

        f(0,3) = neighborPts[i].a1d;
        f(0,4) = neighborPts[i].a2d;
        f(0,5) = neighborPts[i].a3d;

        f(0,6) = neighborPts[i].r;
        f(0,7) = neighborPts[i].g;
        f(0,8) = neighborPts[i].b;

        f(0,9) = neighborPts[i].curvature;

        covariance += (f-u).transpose() * ( f-u )  / ( num -1 );
//        cout<<"covariance : "<<covariance<<endl;
//        cout<<"f : "<<f<<endl;

    }
}

bool  descriptor_geometry_structure_color::descriptor_GSC_extraction(const pcXYZRGBAptr &inputCloud, pcl::PointCloud<pcl::Normal> &normals,int keyptID, vector<neighborPt> &neighborPts)
{
    vector<neighborPt>().swap(neighborPts);
    float searchRadius = 0.1 ;
    int searchNum = 100;

    vector<int> searchId;
    vector<float> distance;

    vector<int>().swap(searchId);
    vector<float>().swap(distance);

    Vector3f keypointNormal;
    keypointNormal(0,0) = normals.points[keyptID].normal_x;
    keypointNormal(1,0) = normals.points[keyptID].normal_y;
    keypointNormal(2,0) = normals.points[keyptID].normal_z;
//    cout<<"keypoint Normal :"<<keypointNormal<<endl;

    pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdTreeFLANN;
    kdTreeFLANN.setInputCloud(inputCloud);

//    kdTreeFLANN.radiusSearch(keyptID, searchRadius, searchId,distance);
    kdTreeFLANN.nearestKSearch(keyptID,searchNum,searchId,distance);
//    cout<<"Neighbot points of keypoint :"<<searchId.size()<<endl;

    double max_alpha=0.0;
    double min_alpha=360.0;

    double max_beta=0.0;
    double min_beta=360.0;

    double max_gama=0.0;
    double min_gama=360.0;

    for(int i=1 ; i<searchId.size() ; ++i)
    {
//        cout<<"======================================="<<endl;
//        cout<<"search ID "<<searchId[i]<<endl;
        neighborPt neighborPt1;
        neighborPt1.r = inputCloud->points[searchId[i]].r / 255.0;
        neighborPt1.g = inputCloud->points[searchId[i]].g / 255.0;
        neighborPt1.b = inputCloud->points[searchId[i]].b / 255.0;
//        cout<<"RGB after normalization : "<<neighborPt1.r<<"  "<<neighborPt1.g<<"  "<<neighborPt1.b<<endl;

        Vector3f neighborPtNormal;
        neighborPtNormal(0,0) = normals.points[searchId[i]].normal_x;
        neighborPtNormal(1,0) = normals.points[searchId[i]].normal_y;
        neighborPtNormal(2,0) = normals.points[searchId[i]].normal_z;
//        cout<<"neighbor point normal :"<<a<<endl;

        Vector3f keypoint2neighbotPt;
        keypoint2neighbotPt(0,0) = inputCloud->points[searchId[i]].x - inputCloud->points[keyptID].x;
        keypoint2neighbotPt(1,0) = inputCloud->points[searchId[i]].y - inputCloud->points[keyptID].y;
        keypoint2neighbotPt(2,0) = inputCloud->points[searchId[i]].z - inputCloud->points[keyptID].z;
//        cout<<"vector from keypoint to neighbot point :"<<keypoint2neighbotPt<<endl;

        neighborPt1.alpha = angle_of_vector(keypointNormal,keypoint2neighbotPt);
        if(neighborPt1.alpha > max_alpha)
            max_alpha = neighborPt1.alpha;
        if(neighborPt1.alpha < min_alpha)
            min_alpha = neighborPt1.alpha;

        neighborPt1.beta = angle_of_vector(neighborPtNormal,keypoint2neighbotPt*(-1));
        if(neighborPt1.beta > max_beta)
            max_beta = neighborPt1.beta;
        if(neighborPt1.beta < min_beta)
            min_beta = neighborPt1.beta;

        neighborPt1.gama = angle_of_vector(keypointNormal,neighborPtNormal);
        if(neighborPt1.gama > max_gama)
            max_gama = neighborPt1.gama;
        if(neighborPt1.gama < min_gama)
            min_gama = neighborPt1.gama;
//        cout<<"angles : "<<neighborPt1.alpha<<"  "<<neighborPt1.beta<<"  "<<neighborPt1.gama<<endl;

        vector<int> searchId_neighbor;
        vector<float> distance_neighbor;
        vector<int>().swap(searchId_neighbor);
        vector<float>().swap(distance_neighbor);
        kdTreeFLANN.nearestKSearch(searchId[i], searchNum, searchId_neighbor, distance_neighbor);
//        kdTreeFLANN.radiusSearch(searchId[i], searchRadius, searchId_neighbor, distance_neighbor);
//        cout<<"Neighborhood pts size: "<<searchId_neighbor.size()<<endl;

        pcaComponentAnalysis::pcaFeature pcaFeature1;
        pcaFeature(inputCloud,pcaFeature1,searchId_neighbor);

        neighborPt1.curvature = pcaFeature1.curvature;
//        cout<<"curvature : "<<pcaFeature1.curvature<<endl;
        lamda_ordered(pcaFeature1.values.lamda1, pcaFeature1.values.lamda2, pcaFeature1.values.lamda3);
//        cout<<"lamda(ordered) : "<<endl;
//        cout<<pcaFeature1.values.lamda1<<"   "<<pcaFeature1.values.lamda2<<"   "<<pcaFeature1.values.lamda3<<endl;

        neighborPt1.a1d = (pcaFeature1.values.lamda1 - pcaFeature1.values.lamda2)/ pcaFeature1.values.lamda1;
        neighborPt1.a2d = (pcaFeature1.values.lamda2 - pcaFeature1.values.lamda3)/ pcaFeature1.values.lamda1;
        neighborPt1.a3d = pcaFeature1.values.lamda3 / pcaFeature1.values.lamda1;

        neighborPts.push_back(neighborPt1);

    }

    for(int i=0 ; i<searchId.size()-1 ; ++i)
    {
        neighborPts[i].alpha = (neighborPts[i].alpha - min_alpha)/(max_alpha - min_alpha);
        neighborPts[i].beta =  (neighborPts[i].beta -  min_beta) /(max_beta -  min_beta);
        neighborPts[i].gama =  (neighborPts[i].gama -  min_gama) /(max_gama -  min_gama);
//        cout<<"angles after normalization : "<<neighborPts[i].alpha<<"  "<<neighborPts[i].beta<<"  "<<neighborPts[i].gama<<endl;
    }

}

