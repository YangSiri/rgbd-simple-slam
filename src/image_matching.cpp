//
// Created by tribbiani on 18-3-24.
// 利用BOW从关键帧中选取候选闭环
//
#include "slamBase.h"
#include "DBoW3/DBoW3.h"

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>

#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

void mergeImg(Mat & dst,Mat &src1,Mat &src2);

FRAME readFrame( string index, ParameterReader& pd );
FRAME readFrame( string rgb_index, string depth_index, string datapath );

bool keyframeSelect(int m,int n, vector<string> &KeyframeID,int treshold,string file);

///mian()
int main(int argc, char** argv)
{
    //initModule_nonfree();

    string parfile ="/home/tribbiani/rgbd-slam/parameters _of_lab.txt";
    ParameterReader pd(parfile);

    string datapath = pd.getData("datapath");
    int end_Index = atoi(pd.getData("end_index").c_str());
    int start_index = atoi(pd.getData("start_index").c_str());

    int thresh = 13;///关键帧相似性阈值设置
    vector<string> keyframeID;
    keyframeSelect(start_index, end_Index, keyframeID, thresh, parfile);

    vector<FRAME> keyframes;

    cout<<"========================="<<endl;
    cout<<"detecting features ..."<<endl;
    //Ptr< FeatureDetector >detector =  FeatureDetector::create("SURF");
    OrbFeatureDetector detector;

    OrbDescriptorExtractor desp;
    //Ptr<DescriptorExtractor> desp = DescriptorExtractor::create("SURF");

    for(int i=0 ; i<keyframeID.size() ; i++)
    {
        keyframes.push_back(readFrame(keyframeID[i],pd));

        detector.detect(keyframes[i].rgb, keyframes[i].kp);
        //detector->detect(image, frames[j-2].kp);

        ///依据keypoint.response筛选最优关键点
        /*
//        vector<KeyPoint> keypoints(frames[j-2].kp);
//        float res_max = 0;
//        float res_min = 1;
//        for (int i = 0; i < keypoints.size(); i++)
//        {
//
//            if (keypoints[i].response > res_max)
//                res_max = keypoints[i].response;
//            if (keypoints[i].response < res_min)
//                res_min = keypoints[i].response;
//        }
//
//        for (int i = 0; i < keypoints.size(); i++)
//        {
//
//            if (keypoints[i].response < res_max)
//                keypoints.erase(keypoints.begin() + i);
//
//        }
        //cout<<"Keypoints : "<<keypoints.size()<<endl;
*/

        desp.compute(keyframes[i].rgb, keyframes[i].kp, keyframes[i].desp);
        //desp->compute(image, keypoints, descriptor);
    }

    ///利用BOW数据库寻找候选影像
    cout<<"============================="<<endl;
    cout<<"Comparing keyframe with database... "<<endl;

    DBoW3::Vocabulary vocab (datapath+"ORBvocabulary.yml.gz");
    if(vocab.empty())
    {
        cout<<"Empty vocabulary!"<<endl;
        return 0;
    }

    int num = keyframeID.size();
    DBoW3::Database db(vocab,true,2);

    for(int i=0 ; i<num ; i++)
        db.add(keyframes[i].desp);
    cout<<"database info : "<<db<<endl;

    for(int i=0 ; i<num ; i++)
    {
        DBoW3::QueryResults query;
        db.query(keyframes[i].desp, query, 3);
        cout<<"\nSearching for keyframe "<<keyframeID[i]<<endl;
        cout<<"<1>, "<<keyframeID[query[1].Id]<<", score "<<query[1].Score<<endl;
        cout<<"<2>, "<<keyframeID[query[2].Id]<<", score "<<query[2].Score<<endl;

        cv::Mat original = imread(datapath+"rgb/"+keyframeID[i]+"_rgb.png");
//        cv::Mat original = imread(datapath+"rgb/"+keyframeID[i]+".png");
        cv::imwrite(datapath+"rgb/"+"keyframe"+to_string(i+1)+".png",original);

        cv::Mat result = imread(datapath+"rgb/"+keyframeID[query[1].Id]+"_rgb.png");
//        cv::Mat result = imread(datapath+"rgb/"+keyframeID[query[1].Id]+".png");

        cv::Mat img ;
        mergeImg(img,original,result);
        imshow("result",img);
//        cvWaitKey(999999);
    }

}



void mergeImg(Mat & dst, Mat &src1, Mat &src2)
{
    int rows = src1.rows+5+src2.rows;
    int cols = src1.cols+5+src2.cols;
    CV_Assert(src1.type () == src2.type ());
    dst.create (rows, cols, src1.type () );
    src1.copyTo (dst(Rect(0,0,src1.cols,src1.rows)));
    src2.copyTo (dst(Rect(src1.cols+5,0,src2.cols,src2.rows)));
}

FRAME readFrame( string index, ParameterReader& pd )
{
    FRAME f;
    string datapath = pd.getData("datapath");
    string rgbpath = datapath+"rgb/";
    string depthpath = datapath+"depth/";

    stringstream ss;
    ss<<rgbpath<<index<<"_rgb.png";
//    ss<<rgbpath<<index<<".png";
    string filename;
    ss>>filename;
    f.rgb = cv::imread( filename );

    ss.clear();
    filename.clear();
//    ss<<depthpath<<index<<".png";
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
//            cv::waitKey(99999);

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
//            cv::waitKey(99999);

            if(num_of_inliers > max_inliers[i])
                max_inliers[i]=num_of_inliers;
            matches_fine.clear();

//            if(num_of_inliers < threshold)
//            {
////                KeyframeID.push_back(i);
//                i=j;
//                keyframe_num++;
//                KeyframeID.push_back(frames[i].frameID);
//                keyframeflag = true;
//                break;
//            }
        }

        ///根据DBOW选取关键帧
        DBoW3::BowVector v1 ;
        vocab.transform(frames[i].desp,v1);
        //cout<<"BoW vector : "<<v1<<endl;
        double score[frames.size()] = {0};
        max_score[i]=0;

        ///寻找最高得分
        for(int j=i+1 ; j<frames.size()-1 ; j++)
        {
            DBoW3::BowVector v2 ;
            vocab.transform(frames[j].desp,v2);
            score[j] = vocab.score(v1,v2);

            if(score[j] > max_score[i])
                max_score[i]=score[j];
            //cout<<"image "<<i<<" vs image "<<j<<" : "<<score<<endl;
        }

        ///利用inliers和BOW得分加权计算相似性similarity
        for(int k=i+1 ; k<frames.size() ; k++)
        {
            double dbowSimi = score[k] / max_score[i] * 100 ;
            double inlierSimi = inliers[k] / max_inliers[i] * 100 ;
            double similarity = dbowSimi * 0.6 + inlierSimi * 0.4;
            //cout<<"similarity between current image "<<k<<" and keyframe "<<i<<" : "<<simi<<endl;

            ///keyframe similarity selection threshold
            if(similarity < threshold)
            {
//                KeyframeID.push_back(i);
                i=k;
                keyframe_num++;
                KeyframeID.push_back(frames[i].frameID);
                keyframeflag = true;
                break;
            }
        }

        if(!keyframeflag)
        {  i++;   continue;  }

        cout<<"Keyframe nums: "<<keyframe_num<<endl;
    }
    return true;
}

