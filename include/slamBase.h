#pragma once

#include <fstream>
#include <vector>
#include <map>
using namespace std;

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>


#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>


#include<Eigen/Core>
#include<Eigen/Geometry>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

struct CAMERA_INTRINSIC_PARAMETERS
{
    double cx,cy,fx,fy,scale;
};

PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera);

cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera);


//chapt 4
struct FRAME
{
    string frameID;
//    int frameID;
    cv::Mat rgb,depth;
    cv::Mat desp;
    vector <cv::KeyPoint> kp;
};

struct RESULT_OF_PNP
{
    cv::Mat rvec, tvec;
    int inliers, flag ;
};

void computeKeyPointAndDesp( FRAME& frame, string detector, string descriptor );

RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera);

class ParameterReader
{
public:
    ParameterReader(string filename)
    {
        ifstream fin( filename.c_str() );
        if(!fin)
        {
            cerr<<"parameter file does not exist."<<endl;
            return;
        }
        while(!fin.eof())
        {
            string str;
            getline(fin,str);
            if(str[0] == '#')
                continue;

            int pos = str.find("=");
            if (pos == -1)
                continue;
            string key = str.substr(0, pos);
            string value = str.substr( pos+1, str.length() );
            data[key] = value;

            if( !fin.good() )
                break;
        }
    }

    ParameterReader()
    {
        string filename = "/home/tribbiani/rgbd-slam/parameters.txt";
        ifstream fin( filename.c_str() );
        if(!fin)
        {
            cerr<<"parameter file does not exist."<<endl;
            return;
        }
        while(!fin.eof())
        {
            string str;
            getline(fin,str);
            if(str[0] == '#')
                continue;

            int pos = str.find("=");
            if (pos == -1)
                continue;
            string key = str.substr(0, pos);
            string value = str.substr( pos+1, str.length() );
            data[key] = value;

            if( !fin.good() )
                break;
        }
    }

    string getData (string key)
    {
        map<string,string>::iterator iter = data.find(key);
        if ( iter == data.end() )
        {
            cerr<<"Parameter name "<<key<<" not found!"<<endl;
            return string("NOT_FOUND");
        }
        return iter->second;
    }

public:
    map<string,string> data;
};

//chap 6
Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec );

PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d& T, CAMERA_INTRINSIC_PARAMETERS& camera);

inline static CAMERA_INTRINSIC_PARAMETERS getDefaultCamera()
{
    ParameterReader pd;
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof( pd.getData( "camera.fx" ).c_str());
    camera.fy = atof( pd.getData( "camera.fy" ).c_str());
    camera.cx = atof( pd.getData( "camera.cx" ).c_str());
    camera.cy = atof( pd.getData( "camera.cy" ).c_str());
    camera.scale = atof( pd.getData( "camera.scale" ).c_str() );
    return camera;
}

inline static CAMERA_INTRINSIC_PARAMETERS getCameraParametersFromFile(string filename)
{
    ParameterReader pd(filename);
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof( pd.getData( "camera.fx" ).c_str());
    camera.fy = atof( pd.getData( "camera.fy" ).c_str());
    camera.cx = atof( pd.getData( "camera.cx" ).c_str());
    camera.cy = atof( pd.getData( "camera.cy" ).c_str());
    camera.scale = atof( pd.getData( "camera.scale" ).c_str() );
    return camera;
}

PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, PointCloud::Ptr newCloud, Eigen::Isometry3d& T, CAMERA_INTRINSIC_PARAMETERS& camera);
