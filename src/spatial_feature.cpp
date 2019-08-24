//
// Created by tribbiani on 18-4-4.
//
#include "slamBase.h"
#include "pca_analysis.h"
#include "keypointExtraction.h"
#include "DBoW3/DBoW3.h"
#include "descriptor_geo_struc_color.h"

#include <time.h>
#include <math.h>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <opencv2/core/eigen.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/features/esf.h>
#include <pcl/features/spin_image.h>
#include <pcl/features/3dsc.h>

using namespace std;
using namespace keypoint;
using namespace pcl;
using namespace cv;

///methods declaration
FRAME readFrame( string index, ParameterReader& pd );
FRAME readFrame (string indexRGB,string indexDepth, string datapath);

/*
void  normal_cal(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,
                 pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree,
                 pcl::PointCloud<pcl::Normal>::Ptr &normals);

void checkNormals(pcl::PointCloud<pcl::Normal>::Ptr normals);

void keypoints_cal(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,pcl::PointCloud<pcl::Normal>::Ptr normal,
                   pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints);

double compute_resolution(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud);

void pfh_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud,
                 pcl::PointCloud<pcl::Normal>::Ptr normal,
                 pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_descriptor);


void esf_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,pcl::PointCloud<pcl::PointXYZ>::Ptr esf_features );

void sc3d_compute(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr input_cloud, pcl::PointCloud<pcl::ShapeContext1980>::Ptr sc3d_features);
*/

bool keyframeSelect(int m,int n, vector<string> &KeyframeID,int treshold,string file);///m:startIndex, n:endIndex

double hausdorffDistance(vector<Eigen::MatrixXf> &a, vector<Eigen::MatrixXf> &b);


///main funtion
int main(int argc, char **argv)
{
    string file ="/home/tribbiani/rgbd-slam/parameters _of_lab.txt";
    ParameterReader pd(file);
    string datapath = pd.getData("datapath");
    int end_Index = atoi(pd.getData("end_index").c_str());
    int start_index = atoi(pd.getData("start_index").c_str());

    int thresh = 25;///关键帧相似性阈值设置
    vector<string> keyframeID;
    keyframeSelect(start_index,end_Index, keyframeID,thresh,file);

    CAMERA_INTRINSIC_PARAMETERS camera = getCameraParametersFromFile(file);
    vector<FRAME> frames;
//    vector<cv::Mat> spatialDescriptors;
    vector<vector<Eigen::MatrixXf>> spatialDescriptors;
    vector<vector<Eigen::MatrixXf>> ().swap(spatialDescriptors);

    for(int i=0 ; i<keyframeID.size() ; i++)
//    for(int i=0 ; i<2 ; i++)
    {
        clock_t startTime, endTime;
        startTime = clock();
        cout<<"--------------------------------"<<endl;
        cout<<"Reading keyframe "<<keyframeID[i]<<" ..."<<endl;

        ///load pcd file
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
        if(pcl::io::loadPCDFile(datapath+"pointcloud"+keyframeID[i]+".pcd",*cloud) == -1)
            return 0;
//        pcl::visualization::CloudViewer viewer("Point Cloud");
//        viewer.showCloud(cloud);
//        while (!viewer.wasStopped())
//        {}

        ///load pointcloud from frame
//        frames.push_back(readFrame(keyframeID[i],pd));
//        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud = image2PointCloud(frames[i].rgb, frames[i].depth, camera);
//        cout<<"cloud size : "<<cloud->points.size()<<endl;

//        pcl::io::savePCDFile("./data/room1.pcd",*cloud);
//        pcl::PLYWriter plyWriter;
//        plyWriter.write();

//        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGBA>());
//        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>()) ;
        //normal_cal(cloud, kdtree, normals);
        //checkNormals(normals);
        //pcl::io::savePCDFile("./data/normals.pcd",normals);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>());
        copyPointCloud(*cloud, *cloud_xyz);
//        viewer.showCloud(cloud_xyz);
//        while (!viewer.wasStopped())
//        {}
        ///依据curvate提取关键点
        keypointOption keypointOption1;
        keypointOption1.radius_featureCalculation = 0.4f;
        keypointOption1.ratioMax = 0.99f;
        keypointOption1.minPtNum = 20;
        keypointOption1.radiusNonMax = 0.4f;
        keypointDetection keypointDetection1(keypointOption1);

        pcl::PointIndicesPtr keypointIndices;
        keypointDetection1.keypointDetectionBasedonCurvature(cloud_xyz,keypointIndices);
        cout<<"Keypoints : "<<keypointIndices->indices.size()<<endl;

        ///可视化关键点点云
//        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr keypointsCloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
//        for(size_t i=0 ; i<keypointIndices->indices.size() ; ++i)
//            keypointsCloud->push_back(cloud->points[keypointIndices->indices[i]]);
//        pcl::visualization::CloudViewer viewer("keypoints");
//        viewer.showCloud(keypointsCloud);
//        viewer.showCloud(cloud_xyz);
//        while (!viewer.wasStopped())
//        {
//
//        }

        ///descirptor of geometry, structure, color, curvature

        pcl::PointCloud<pcl::Normal> normals;
        pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> normalEstimation;
        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGBA>);

        normalEstimation.setSearchMethod(kdtree);
        normalEstimation.setInputCloud(cloud);
        normalEstimation.setKSearch(10);
        normalEstimation.compute(normals);

        descriptor_geometry_structure_color desp_GSC;///构建对象
        vector<Eigen::MatrixXf> covs;
        vector<Eigen::MatrixXf>().swap(covs);

        for(int i=0 ; i<keypointIndices->indices.size() ; ++i)
        {
            vector<descriptor_geometry_structure_color::neighborPt> neighborPts;
            vector<descriptor_geometry_structure_color::neighborPt>().swap(neighborPts);
            desp_GSC.descriptor_GSC_extraction(cloud, normals,keypointIndices->indices[i], neighborPts);

            Eigen::MatrixXf u(1,10);///描述向量均值
            u = Eigen::MatrixXf::Zero(1,10);
            desp_GSC.averageDescriptor(neighborPts,u);
//            cout<<"average "<<i<<": "<<u<<endl;

            Eigen::MatrixXf covariance(10,10) ;///关键点处协方差矩阵
            covariance = Eigen::MatrixXf::Zero(10,10);
            desp_GSC.covarianceEstablish(neighborPts,u,covariance);

//            cv::Mat cov;
//            eigen2cv(covariance,cov);
//            cout<<"Cov "<<i<<" : "<<covariance<<endl;

//            SelfAdjointEigenSolver<MatrixXf> eigenSolver(covariance);
//            cv::Mat temp(1,10,CV_32FC1);
//            Eigen::MatrixXf eigenValues = eigenSolver.eigenvalues().transpose();
//            eigen2cv( eigenValues ,temp);
            covs.push_back(covariance);
//            cout<<"temp "<<i<<" : "<<eigenValues <<endl;

//            cout<<"vectors :"<<eigenSolver.eigenvectors()<<endl;
//            cout<<"values :"<<eigenSolver.eigenvalues()<<endl;
        }
        spatialDescriptors.push_back(covs);

        ///计算每个关键点的LCF
//        keypointNeiborAnalysis keypointNeiborAnalysis1;
////        vector<keypointNeiborAnalysis::neiborPoint> neiborPoints;
//
//        double neigborRadius = 0.5 ;//邻域点搜索半径
//        int num = 10;//环数
//        vector<keypointNeiborAnalysis::localReferFrame> localReferFrames;
//
//        keypointNeiborAnalysis1.calculateNeiborPoints(cloud_xyz,keypointIndices,localReferFrames,neigborRadius,num);
////        cout<<"Neibor points size : "<<neiborPoints.size()<<endl;
////        keypointNeiborAnalysis1.solveLocalreferFrame(neiborPoints,localReferFrames);
////        cout<<"LCF : "<<localReferFrames.size()<<endl

     /*
        ///将平面划分为以关键点为中心的正方形网格
        int gridNums = 25;
        double gridLength = neigborRadius / gridNums ;
        cv::Mat spatialDescriptor(localReferFrames.size(),9,CV_32FC1);
        pcl::PointXYZ coordinate;

        for(int i=0 ; i<localReferFrames.size() ; ++i)
        {
            ///分别计算三个特征向量的网格坐标
            int row = 0;
            for(int j=0 ; j<3 ; ++j)
            {
//                float r = localReferFrames[i].eVec.at<float>(j,0) * localReferFrames[i].eVec.at<float>(j,0) +
//                           localReferFrames[i].eVec.at<float>(j,1) * localReferFrames[i].eVec.at<float>(j,1) +
//                           localReferFrames[i].eVec.at<float>(j,2) * localReferFrames[i].eVec.at<float>(j,2);
                float ratio = sqrt( localReferFrames[i].lamda.at<float>(j,0)*localReferFrames[i].lamda.at<float>(j,0));
                coordinate.x = localReferFrames[i].eVec.at<float>(j,0) * ratio ;
                coordinate.y = localReferFrames[i].eVec.at<float>(j,1) * ratio ;
                coordinate.z = localReferFrames[i].eVec.at<float>(j,2) * ratio ;

                int x = (int)(coordinate.x / gridLength + coordinate.x/fabs(coordinate.x));
                int y = (int)(coordinate.y / gridLength + coordinate.y/fabs(coordinate.y));
                int z = (int)(coordinate.z / gridLength + coordinate.z/fabs(coordinate.z));
                spatialDescriptor.at<float>(i,row) =(float) x ;
                row++;
                spatialDescriptor.at<float>(i,row) =(float) y ;
                row++;
                spatialDescriptor.at<float>(i,row) =(float) z ;
                row++;
            }

        }
*/

//        cv::Mat spatialDescriptor(localReferFrames.size(),num,CV_32FC1);
//        for(int i=0 ; i<localReferFrames.size() ; ++i)
//        {
//            for (int j = 1; j <= num; ++j)
//                spatialDescriptor.at<float>(i, j-1) = int(localReferFrames[i].score[j]);
//        }
//        cout<<spatialDescriptor<<endl;
//        spatialDescriptors.push_back(spatialDescriptor);

/*
//        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>());
//        keypoints_cal(cloud,normals,kdtree,keypoints);
//        pcl::PointCloud<pcl::Normal>::Ptr key_normals(new pcl::PointCloud<pcl::Normal>()) ;
//
//        ///create a PointCloud(PointXYZRGBA) of keypoints(PointXYZ)
//        PointCloud::Ptr keypoints_cloud (new PointCloud());
//        pcl::copyPointCloud(*keypoints,*keypoints_cloud);
//
//        normal_cal(keypoints_cloud,kdtree,key_normals);

//        pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_descriptors(new pcl::PointCloud<pcl::PFHSignature125>());
//        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
//        pcl::copyPointCloud(*cloud, *cloud_xyz);
//        pfh_compute(cloud_xyz, normals, pfh_descriptors);

//        pcl::PointCloud<pcl::PointXYZ>::Ptr esf_descriptors (new pcl::PointCloud<pcl::PointXYZ>());
//        esf_compute(keypoints,esf_descriptors);
        */

        endTime = clock();
        cout<<"Frame "<<keyframeID[i]<<" finished. Time :"<<(double)(endTime-startTime)/CLOCKS_PER_SEC<<" s"<<endl;
        cout<<"=================================="<<endl;

    }

    for(size_t i =0 ; i<spatialDescriptors.size() ; ++i)
        for(size_t j =i+1 ; j<spatialDescriptors.size() ; ++j)
        {
            double hausdorffDis = hausdorffDistance(spatialDescriptors[i],spatialDescriptors[j]);
            cout<<"image "<<keyframeID[i] <<" vs image "<<keyframeID[j]<<" : "<<hausdorffDis<<endl;
        }


    ///利用DBOW计算两帧图像的相似性
//    cout<<"Frames all done. Creating spatial vocabulary ..."<<endl;
//    DBoW3::Vocabulary spatialVocab;
//    spatialVocab.create(spatialDescriptors);
//    if(spatialVocab.empty())
//    {
//        cout<<"Empty spatial vocabulary!"<<endl;
//        return 0;
//    }
//    cout<<"spatial vocal info :"<<spatialVocab<<endl;
////    spatialVocab.save("spatialVocabulary.yml.gz");
//
//    for(size_t i =0 ; i<spatialDescriptors.size() ; ++i)
//    {
//        DBoW3::BowVector v1,v2;
//        spatialVocab.transform(spatialDescriptors[i],v1);
//        cout<<"----------------------------------"<<endl;
//
//        for(size_t j =i+1 ; j<spatialDescriptors.size() ; ++j)
//        {
//            spatialVocab.transform(spatialDescriptors[j],v2);
//            double score = spatialVocab.score(v1,v2);
//            cout<<"image "<<keyframeID[i] <<" vs image "<<keyframeID[j]<<" : "<<score<<endl;
//        }
//
//    }

}

FRAME readFrame( string index, ParameterReader& pd )
{
    FRAME f;
    string datapath = pd.getData("datapath");
    string rgbpath = datapath+"rgb/";
    string depthpath = datapath+"depth/";

    stringstream ss;
    ss<<rgbpath<<index<<"_rgb.png";
    string filename;
    ss>>filename;
    f.rgb = cv::imread( filename );

    ss.clear();
    filename.clear();
    ss<<depthpath<<index<<"_depth.png";
    ss>>filename;

    f.depth = cv::imread( filename, -1 );
    f.frameID = index;
    return f;
}

FRAME readFrame (string indexRGB,string indexDepth, string datapath)
{
    FRAME f;

    stringstream ss;
    string filename;

    ss<<datapath<<indexRGB;
    ss>>filename;
    f.rgb = cv::imread (filename);

    ss.clear();
    filename.clear();

    ss<<datapath<<indexDepth;
    ss>>filename;
    f.depth = cv::imread(filename,CV_LOAD_IMAGE_ANYCOLOR|CV_LOAD_IMAGE_ANYDEPTH);

    return f;
}

///利用inliers选择keyframe
bool keyframeSelect(int m,int n, vector<string> &KeyframeID,int threshold,string file)
{
    cout<<"========================="<<endl;
    cout<<"Selecting keyframes ..."<<endl;
    cv::OrbFeatureDetector detector;
    cv::OrbDescriptorExtractor despExtract;
    vector<FRAME> frames;
    vector<cv::Mat> descriptors;
    ParameterReader pd(file);
    string datapath = pd.getData("datapath");

    cout<<"reading data and compute descriptors."<<endl;
    ifstream fin(datapath+"associate.txt");
    if(!fin)
    {
        cout<<"associate file doesn't exist!"<<endl;
        return 0;
    }
    ///读取数据
    int dataNums=0;
    for(int i=m ; i<n ; ++i)
    {

        if(fin.eof())
        {
            cout<<"End of file!"<<endl;
            break;
        }
        dataNums++;

        string line;
        if(dataNums == 1)
        {
            for(int j=1 ; j<i ; ++j)
                getline(fin,line);
        }
        getline(fin,line);
        stringstream ss (line);
        string rgb, rgbpath, depth, depthpath;
        ss>>rgb;
        ss>>rgbpath;
        ss>>depth;
        ss>>depthpath;
//        cout<<rgbpath<<endl;

        FRAME tempframe =readFrame(rgbpath, depthpath, datapath);
        tempframe.frameID = rgb;
        cv::Mat temprgb = tempframe.rgb;
        cv::Mat tempdepth = tempframe.depth;

        detector.detect(tempframe.rgb, tempframe.kp);
        despExtract.compute(tempframe.rgb, tempframe.kp, tempframe.desp);
        descriptors.push_back(tempframe.desp);

        frames.push_back (tempframe);
        cout<<"frame "<<i<<" done."<<endl;
    }

    cout<<"creating vocabulary. "<<endl;
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    vocab.save(datapath+"ORBvocabulary.yml.gz");

    ///select inliers
    cv::BFMatcher matcher;
    vector<cv::DMatch> matches;
    vector<cv::DMatch> matches_fine;
    int keyframe_num = 0;//number of keyframes
    double max_score[frames.size()];
    double max_inliers[frames.size()];

    KeyframeID.push_back(frames[0].frameID);

    for(int i=0 ; i<frames.size()-1 ; )
    {
        cout<<"\nKeyframe No:"<<frames[i].frameID<<endl;
//        cout<<"\nKeyframe No:"<<(i+m)<<endl;
        bool keyframeflag = false;

        ///根据inliers选取关键帧
        int inliers[frames.size()] = {0};
        max_inliers[i]=0;

        for (int j=i+1 ; j<frames.size()-1 ; j++)
        {
            matcher.match(frames[i].desp, frames[j].desp, matches);

//            cv::Mat imgWin;
//            cv::drawMatches(frames[i].rgb, frames[i].kp, frames[j].rgb, frames[j].kp, matches, imgWin);
//            cv::imshow("Matches", imgWin);
//            cv::waitKey(99);

            Mat fundmental;
            vector<uchar> ransac_status;

            Mat p1(matches.size(), 2, CV_32F);
            Mat p2(matches.size(), 2, CV_32F);
            for (int k = 0; k < matches.size(); k++)
            {
                p1.at<float>(k, 0) = frames[i].kp[matches[k].queryIdx].pt.x;
                p1.at<float>(k, 1) = frames[i].kp[matches[k].queryIdx].pt.y;
                p2.at<float>(k, 0) = frames[j].kp[matches[k].trainIdx].pt.x;
                p2.at<float>(k, 1) = frames[j].kp[matches[k].trainIdx].pt.y;
            }

            fundmental = findFundamentalMat(p1, p2, FM_RANSAC, 3, 0.99, ransac_status);
//            cout<<"RANSAC result : "<< ransac_status.size()<<endl;

            int num_of_inliers = 0;
            for (int k = 0; k < ransac_status.size(); k++)
            {
                if (ransac_status[k] == 1)
                {
                    matches_fine.push_back(matches[k]);
                    num_of_inliers++;
                }
            }
            inliers[j]=num_of_inliers;

//            cv::drawMatches(frames[i].rgb, frames[i].kp, frames[j].rgb, frames[j].kp, matches_fine, imgWin);
//            cv::imshow("Inliers", imgWin);
//            cv::waitKey(99);

            if(num_of_inliers > max_inliers[i])
                max_inliers[i]=num_of_inliers;
            matches_fine.clear();

            if(num_of_inliers < threshold)
            {
//                KeyframeID.push_back(i);
                i=j;
                keyframe_num++;
                KeyframeID.push_back(frames[i].frameID);
                keyframeflag = true;
                break;
            }
        }

//        ///根据DBOW选取关键帧
//        DBoW3::BowVector v1 ;
//        vocab.transform(frames[i].desp,v1);
//        //cout<<"BoW vector : "<<v1<<endl;
//        double score[frames.size()] = {0};
//        max_score[i]=0;
//
//        ///寻找最高得分
//        for(int j=i+1 ; j<frames.size()-1 ; j++)
//        {
//            DBoW3::BowVector v2 ;
//            vocab.transform(frames[j].desp,v2);
//            score[j] = vocab.score(v1,v2);
//
//            if(score[j] > max_score[i])
//                max_score[i]=score[j];
//            //cout<<"image "<<i<<" vs image "<<j<<" : "<<score<<endl;
//        }
//
//        ///利用inliers和BOW得分加权计算相似性similarity
//        for(int k=i+1 ; k<frames.size() ; k++)
//        {
//            double dbowSimi = score[k] / max_score[i] * 100 ;
//            double inlierSimi = inliers[k] / max_inliers[i] * 100 ;
//            double similarity = dbowSimi * 0.6 + inlierSimi * 0.4;
//            //cout<<"similarity between current image "<<k<<" and keyframe "<<i<<" : "<<simi<<endl;
//
//            ///keyframe similarity selection threshold
//            if(similarity < treshold)
//            {
////                KeyframeID.push_back(i);
//                i=k;
//                keyframe_num++;
//                KeyframeID.push_back(frames[i].frameID);
//                keyframeflag = true;
//                break;
//            }
//        }

        if(!keyframeflag)
        {  i++;   continue;  }

        cout<<"Keyframe nums: "<<keyframe_num<<endl;
    }
    return true;
}


double hausdorffDistance(vector<Eigen::MatrixXf> &a, vector<Eigen::MatrixXf> &b)
{
    double min_eucliDis [a.size()];
    double hausdorff = 0.0;
    for(size_t i=0 ; i<a.size() ; ++i)
    {
        min_eucliDis[i] = 999.0;

        for(size_t j=0 ; j<b.size() ;++j)
        {
            Eigen::MatrixXf corr (10,10);
            corr = a[i].inverse() * b[j];
//            cout<<"corr : "<<corr<<endl;

            SelfAdjointEigenSolver<MatrixXf> eigenSolver(corr);
            Eigen::MatrixXf eigenValues = eigenSolver.eigenvalues().transpose();
//            cout<<"eigenvalues"<<i<<" & "<<j<<" : "<<eigenValues<<endl;

            double euclideanDis = 0.0;
//            cout<<a[i]<<"\n"<<b[j]<<endl;
            for(int k=0 ; k<10 ;++k)
                euclideanDis += log( fabs(eigenValues(0,k)) )*log( fabs(eigenValues(0,k)) );
            euclideanDis = sqrt(euclideanDis);

//            double euclideanDis = norm(a[i],b[j],CV_L2);
//            cout<<"Euclidean distance : "<<euclideanDis<<endl;
            if (euclideanDis < min_eucliDis[i])
                min_eucliDis[i] = euclideanDis;
        }
//        cout<<"min distance : "<<min_eucliDis[i]<<endl;
//        if(min_eucliDis[i] > hausdorff && min_eucliDis[i] < 999.0)
//            hausdorff = min_eucliDis[i];
            hausdorff += min_eucliDis[i] ;
    }
    return  hausdorff / a.size();
}


/*
///calculate normals
void  normal_cal(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree,pcl::PointCloud<pcl::Normal>::Ptr &normals)
{

    pcl::NormalEstimation<pcl::PointXYZRGBA,pcl::Normal> normalEstimation;

    normalEstimation.setInputCloud(cloud);
    normalEstimation.setSearchMethod(tree);
    //normalEstimation.setRadiusSearch(0.05);
    normalEstimation.setKSearch(10);
    normalEstimation.compute(*normals);

    ///Moving least square filtering and normals
//    pcl::MovingLeastSquares<pcl::PointXYZRGBA, pcl::PointNormal> mls;
//
//    mls.setComputeNormals(true);
//    mls.setInputCloud(cloud);
//    mls.setPolynomialFit(true);
//    mls.setSearchMethod(tree);
//    mls.setSearchRadius(0.03);
//    mls.process(*pts_normals);
//
//    cout<<"normal size : "<<pts_normals->size()<<endl;
    //pcl::io::savePCDFileASCII("./data/normals.pcd",*normal);

    ///可视化
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normals"));
//    viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, "cloud");
//
//    viewer->addPointCloudNormals<pcl::PointXYZRGBA, pcl::Normal>(cloud, normals, 30, 0.1, "normals");
//    while (!viewer->wasStopped())
//    {
//        viewer->spinOnce(100);
//        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//    }
}

void checkNormals(pcl::PointCloud<pcl::Normal>::Ptr normals)
{
    for (int i = 0; i < normals->size(); ++i)
    {
        if(!pcl::isFinite<pcl::Normal>(normals->points[i]))
        {
            normals->points[i].normal_x = 0.577;
            normals->points[i].normal_y = 0.577;
            normals->points[i].normal_z = 0.577;
            normals->points[i].curvature = 0.0;
        }

        if(isnan(normals->points[i].curvature))
        {
            normals->points[i].normal_x = 0.577;
            normals->points[i].normal_y = 0.577;
            normals->points[i].normal_z = 0.577;
            normals->points[i].curvature = 0.0;
        }
    }
}

///calculate keypoints
void keypoints_cal(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,pcl::PointCloud<pcl::Normal>::Ptr normal, pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints)
{
    ///extract range image from PointCLoud
//    float noise_level =  0.0 ;
//    boost::shared_ptr<pcl::RangeImage> range_image_ptr (new pcl::RangeImage);
//    pcl::RangeImage& rangeImage = *range_image_ptr;
//    rangeImage.createFromPointCloud(cloud);
//    pcl::RangeImageBorderExtractor rangeImageBorderExtractor;
//    pcl::NarfKeypoint narfKeypoint_detector(&rangeImageBorderExtractor);
//    narfKeypoint_detector.setRangeImage(rangeImage);

    pcl::ISSKeypoint3D<pcl::PointXYZRGBA,pcl::PointXYZ,pcl::Normal> issKeypoint3D;
    //boost::shared_ptr<const pcl::PointCloud<pcl::Normal>> constNormal (&normal);

    issKeypoint3D.setNormals(normal);
    issKeypoint3D.setSearchMethod(tree);
    issKeypoint3D.setInputCloud(cloud);

    double res = compute_resolution(cloud);
    cout<<"cloud resolution : "<<res<<endl;
    issKeypoint3D.setNonMaxRadius(res * 4.0 );
    issKeypoint3D.setSalientRadius(res * 6.0 );
    issKeypoint3D.compute(*keypoints);

    cout<<"keypoints size : "<<keypoints->size()<<endl;
    //pcl::io::savePCDFile("./data/keypoints.pcd",keypoints );
}

///calculate cloud resolution
double compute_resolution(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud )
{
    double res = 0.0 ;
    int n_pts = 0;
    int nres;
    const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr const_cloud(cloud);

    vector<int> indices(2);
    vector<float> sqr_distance(2);
    pcl::search::KdTree<pcl::PointXYZRGBA> kdtree;
    kdtree.setInputCloud(const_cloud);

    for(size_t i=0 ; i<const_cloud->size() ; ++i)
    {
        if(!pcl_isfinite((*const_cloud)[i].x))
            continue;
        nres= kdtree.nearestKSearch(i, 2, indices, sqr_distance);
        if(nres == 2)
        {
            res += sqrt(sqr_distance[1]);
            ++n_pts;
        }
    }
    if(n_pts != 0)
        res /= n_pts;
    return res;
}

///extract PFH descriptors
void pfh_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud, pcl::PointCloud<pcl::Normal>::Ptr normal,pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_descriptor)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfhEstimation;

    pfhEstimation.setInputCloud(in_cloud);
    pfhEstimation.setSearchMethod(tree);
    pfhEstimation.setInputNormals(normal);
    pfhEstimation.setRadiusSearch(0.05);//5cm
    pfhEstimation.compute(*pfh_descriptor);

    cout<<"descriptors : "<<pfh_descriptor->size()<<endl;
}

///extract ESF descriptors
void esf_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,pcl::PointCloud<pcl::PointXYZ>::Ptr esf_features )
{
    pcl::ESFEstimation<pcl::PointXYZ, pcl::PointXYZ> esfEstimation;
    esfEstimation.setInputCloud(input_cloud);
    esfEstimation.compute(*esf_features);
}

///extract 3D Shape Context descriptors
void sc3d_compute(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr input_cloud, pcl::PointCloud<pcl::ShapeContext1980>::Ptr sc3d_features )
{
    pcl::ShapeContext3DEstimation<pcl::PointXYZRGBA, pcl::PointXYZ,pcl::ShapeContext1980> sc3dEstimation;
    sc3dEstimation.setInputCloud(input_cloud);
    sc3dEstimation.compute(*sc3d_features);
}

///extract Spin Image descriptors
void spinImage_compute(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr input_cloud, pcl::PointCloud<pcl::Histogram<153>>::Ptr spinImage_features )
{
    pcl::SpinImageEstimation<pcl::PointXYZRGBA, pcl::Normal,pcl::Histogram<153>> spinImageEstimation;
    spinImageEstimation.setInputCloud(input_cloud);
    spinImageEstimation.compute(*spinImage_features);
}
 */