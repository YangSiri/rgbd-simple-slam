//
// Created by tribbiani on 18-4-9.
//

#include "keypointExtraction.h"

#include <pcl/filters/extract_indices.h>

using namespace keypoint;
using namespace std;
using namespace pcl;

bool cmpBasedonCurvature(keypointDetection::pcaFeature &a, keypointDetection::pcaFeature &b)
{
    if(a.curvature > b.curvature)
        return true;
    else
        return false;
}

bool keypointDetection::keypointDetectionBasedonCurvature(const pcXYZptr &inputCloud,
                                                          pcl::PointIndicesPtr &keypointIndices)
{
    vector<pcaFeature> features(inputCloud->size());
    calculatePCAfeaturesOfPointCloud(inputCloud,kpO.radius_featureCalculation,features);

    size_t keypointNum =0 ;

    PointIndicesPtr candidatesIndices(new PointIndices());
    pruneUnstablePoints(features,kpO.ratioMax,candidatesIndices);

    vector<pcaFeature> stableFeatures;
    for(size_t i=0 ; i< candidatesIndices->indices.size() ; ++i)
        stableFeatures.push_back(features[candidatesIndices->indices[i]]);

    PointIndicesPtr  nonMaximaIndices(new PointIndices());
    nonMaxiamaSuppression(stableFeatures,nonMaximaIndices);

    keypointIndices = nonMaximaIndices;
    keypointNum = keypointIndices->indices.size();
    return true;

}

bool keypointDetection::pruneUnstablePoints(const std::vector<pcaFeature> &features, float ratioMax,
                                            pcl::PointIndicesPtr &indicesStable)
{
    for(size_t i= 0 ; i < features.size() ; ++i)
    {
        float ratio1, ratio2;
        ratio1 = features[i].values.lamda2 / features[i].values.lamda1;
        ratio2 = features[i].values.lamda3 / features[i].values.lamda2 ;

        if(ratio1 < ratioMax && ratio2 < ratioMax && features[i].ptnum > kpO.minPtNum)
            indicesStable->indices.push_back( int(i) );
    }
    return true;
}

bool keypointDetection::nonMaxiamaSuppression(std::vector<pcaFeature> &features,//stable features
                                              pcl::PointIndicesPtr &indicesNonMaxSup)
{
    sort(features.begin(),features.end(), cmpBasedonCurvature);
    PointCloud<PointXYZ> pointCloud;

    /*建立UnSegment以及UnSegment的迭代器，存储未分割的点号*/
    set<size_t ,less<size_t >> unvisitedPtId;
    set<size_t ,less<size_t >>::iterator iterUnseg;

    for(size_t i=0 ; i<features.size() ; ++i)
    {
        unvisitedPtId.insert(i);
        pointCloud.points.push_back(features[i].pt);
    }

    pcl::KdTreeFLANN<PointXYZ> kdTreeFLANN;
    kdTreeFLANN.setInputCloud(pointCloud.makeShared());

    vector<int> searchIndices;
    vector<float> distances;

    size_t keypointNum = 0;
    do{
        keypointNum++;
        vector<int> ().swap(searchIndices);
        vector<float>().swap(distances);

        size_t id;
        iterUnseg = unvisitedPtId.begin();
        id = *iterUnseg;
        indicesNonMaxSup->indices.push_back( int(features[id].ptId) );
        unvisitedPtId.erase(id);

        kdTreeFLANN.radiusSearch(features[id].pt, kpO.radiusNonMax, searchIndices, distances);

        for(size_t i=0 ; i<searchIndices.size() ; ++i)
            unvisitedPtId.erase(searchIndices[i]);

    }while(!unvisitedPtId.empty());

    return true;

}