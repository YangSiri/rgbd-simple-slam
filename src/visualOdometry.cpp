#include <pcl/visualization/cloud_viewer.h>
//chaptr 5
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

#include "slamBase.h"

FRAME readFrame(int index, ParameterReader& pd);

FRAME readFrame (string indexRGB,string indexDepth,string datapath);

double normofTransform( cv::Mat rvec, cv::Mat tvec);

void pictureDownSample(cv::Mat img, cv::Mat &dst);

///main()
int main( int argc, char** argv )
{
    string parafile ="/home/tribbiani/rgbd-slam/parameters_of_tum_room.txt";
    ParameterReader pd(parafile);
    int startIndex = atoi( pd.getData("start_index").c_str() );
    int endIndex = atoi( pd.getData("end_index").c_str() );

    ///reading associate.txt
    string datapath = pd.getData("datapath");
    vector<FRAME> frames;
    FRAME zeroFrame;
    frames.push_back(zeroFrame);
    ifstream fin(datapath + "associate.txt");
    if(!fin)
    {
        cout<<"associate file doesn't exist!"<<endl;
        return 0;
    }
    int dataNums=0;
    for(int i=startIndex ; i<endIndex ; ++i)
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
//        cout<<temprgb.rows<<"    "<<temprgb.cols<<endl;
//
//        pictureDownSample(temprgb,tempframe.rgb);
//        cout<<tempframe.rgb.rows<<"     "<<tempframe.rgb.cols<<endl;
//        cv::namedWindow("Down Sample Image");
//        cv::imshow("Down Sample Image",tempframe.depth);
//        cv::waitKey(600);
//        tempdepth.convertTo(tempframe.depth,CV_16UC1);
//        cout<<tempframe.depth<<endl;
        frames.push_back (tempframe);
        cout<<"frame "<<i<<" done."<<endl;
    }

    cout<<"Initializing ..."<<endl;
    int currIndex = startIndex;
    FRAME lastFrame = frames[35];
//    FRAME lastFrame = readFrame( currIndex, pd );

    string detector = pd.getData("detector");
    string descriptor = pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera = getCameraParametersFromFile(parafile);
    computeKeyPointAndDesp( lastFrame, detector, descriptor );

//    PointCloud::Ptr cloud (new PointCloud);
//    if( pcl::io::loadPCDFile(datapath+"pointcloud"+lastFrame.frameID+".pcd", *cloud) == -1)
//    {
//        PCL_ERROR("Couldn't read pcd file!");
//        return 0;
//    }
    PointCloud::Ptr cloud = image2PointCloud( lastFrame.rgb, lastFrame.depth, camera );

    pcl::visualization::CloudViewer viewer("viewer");

    bool visualize = pd.getData("visualize_pointcloud") == string("yes");

    int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    double max_norm = atof( pd.getData("max_norm").c_str() );

    for(currIndex = startIndex+1  ; currIndex<dataNums ; currIndex++)
    {
        cout<<"====================="<<endl;
        cout<<"Reading files "<<currIndex<<endl;
        FRAME currFrame = frames[currIndex];
//        FRAME currFrame = readFrame( currIndex, pd );
        ///point cloud of every frame
//        PointCloud::Ptr framecloud (new PointCloud);
//        if( pcl::io::loadPCDFile(datapath+"pointcloud"+currFrame.frameID+".pcd", *framecloud) == -1)
//        {
//            PCL_ERROR("Couldn't read pcd file!");
//            return 0;
//        }
        PointCloud::Ptr framecloud = image2PointCloud( currFrame.rgb, currFrame.depth, camera );

//        viewer.showCloud(framecloud);
//        while (!viewer.wasStopped())
//        {}

        computeKeyPointAndDesp( currFrame, detector, descriptor );
        //比较currFrame和lastFrame
        RESULT_OF_PNP result = estimateMotion( lastFrame, currFrame, camera );
        if( result.inliers < min_inliers || result.flag == 0)//inliers不够，放弃该帧
            continue;

        double norm = normofTransform(result.rvec, result.tvec);
        cout<<"norm = "<<norm<<"\n"<<endl;
        if( norm > max_norm )
            continue;

        Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec );
        cout<<"T= "<<T.matrix()<<endl;

        cloud = joinPointCloud( cloud, framecloud, T, camera );

        if( visualize == true )
            viewer.showCloud (cloud);

        lastFrame = currFrame;
    }
    pcl::io::savePCDFile( "./data/keyframecloud.pcd", *cloud );
    return 0;
}


FRAME readFrame (int index, ParameterReader& pd)
{
    FRAME f;
    string rgbDir = pd.getData("rgb_dir");
    string depthDir = pd.getData("depth_dir");

    string rgbExt = pd.getData("rgb_extension");
    string depthExt = pd.getData("depth_extension");

    stringstream ss;
    string filename;

    ss<<rgbDir<<index<<rgbExt;
    ss>>filename;
    f.rgb = cv::imread (filename);

    ss.clear();
    filename.clear();

    ss<<depthDir<<index<<depthExt;
    ss>>filename;
    f.depth = cv::imread(filename,-1);

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

double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
    return fabs( min( cv::norm(rvec), 2*M_PI-cv::norm(rvec) ) ) + fabs( cv::norm(tvec) );
}

void pictureDownSample(cv::Mat img, cv::Mat &dst)
{
    // 降采样高斯权重
    double a = 0.6;
    double w[5] = {1.0/4 - a/2.0, 1.0/4, a, 1.0/4, 1.0/4 - a/2.0};// 这里面double类型需要1.0/4而不是1/4，我有时会疏忽这一点
    // 转换图像数据类型
    cv::Mat src = img.clone();
    src.convertTo(src, CV_64FC1);
    // 定义目标矩阵，矩阵行列数减半向上取整(/2.0而不是/2)
    dst = cv::Mat(512, 414, CV_64FC1);
//    dst = cv::Mat((int)(src.rows / 2.0 + 0.5), (int)(src.cols / 2.0 + 0.5), CV_64FC1, Scalar(0.0));
    // 定义x方向降采样临时矩阵
    cv::Mat temp_x(src.rows, (int)(src.cols / 2.0 + 0.5), CV_64FC1);
    // 定义x方向边界扩充，两边扩充的两列像素直接复制原图像两边缘像素
    cv::Mat paddImg_x(src.rows, src.cols + 4, CV_64FC1);
    for (int i = 0; i < paddImg_x.rows; ++i)
    {
        for (int j = 0; j < paddImg_x.cols; ++j)
        {
            if (j < 2)
                paddImg_x.at<double>(i, j) = src.at<double>(i, j);
            else if (j >= 2 && j < paddImg_x.cols - 2)
                paddImg_x.at<double>(i, j) = src.at<double>(i, j - 2);
            else
                paddImg_x.at<double>(i, j) = paddImg_x.at<double>(i, j - 1);
        }
    }
    // x方向降采样
    for (int i = 0; i < temp_x.rows; ++i)
    {
        for (int j = 0; j < temp_x.cols; ++j)
        {
            for (int m = -2; m <= 2; ++m)
            {
                // 权重和
                temp_x.at<double>(i, j) += w[m + 2] * paddImg_x.at<double>(i, j * 2 + m + 2);
            }
        }
    }
    // 基于已得到的temp_x，定义y方向边界扩充矩阵
    cv::Mat paddImg_y(temp_x.rows + 4, temp_x.cols, CV_64FC1);
    for (int i = 0; i < paddImg_y.rows; ++i)
    {
        for (int j = 0; j < paddImg_y.cols; ++j)
        {
            if (i < 2)
                paddImg_y.at<double>(i, j) = temp_x.at<double>(i, j);
            else if (i >= 2 && i < paddImg_y.rows - 2)
                paddImg_y.at<double>(i, j) = temp_x.at<double>(i - 2, j);
            else
                paddImg_y.at<double>(i, j) = paddImg_y.at<double>(i - 1, j);
        }
    }
    // y方向上的降采样
    for (int i = 0; i < dst.rows; ++i)
    {
        for (int j = 0; j < dst.cols; ++j)
        {
            for (int m = -2; m <= 2; ++m)
            {
                // 权重和
                dst.at<double>(i, j) += w[m + 2] * paddImg_y.at<double>(i * 2 + m + 2, j);
            }
        }
    }
    // 转换图像数据类型
    dst.convertTo(dst, CV_8U);
}
