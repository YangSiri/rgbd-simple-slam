
//chaptr 3
#include "slamBase.h"

PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    PointCloud::Ptr cloud( new PointCloud );

    for (int m=0 ; m < depth.rows; m+=2)
        for (int n=0 ; n < depth.cols ; n+=2)
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
//            cout<<"depth: "<<d<<endl;
            // d 存在值，则向点云增加一个点
            PointT p;

//            cout<<camera.scale<<endl;
//            // 计算这个点的空间坐标
//            p.z = double(d) / camera.scale;
//            p.x = (n - camera.cx) * p.z / camera.fx;
//            p.y = (m - camera.cy) * p.z / camera.fy;
            // 计算这个点的空间坐标
            p.z = static_cast<float>(double(d) / camera.scale);
            p.x = static_cast<float>((n + 0.5 - camera.cx) * p.z / camera.fx);
            p.y = static_cast<float>((m + 0.5 - camera.cy) * p.z / camera.fy);

            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            cloud->points.push_back( p );
        }

    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}

//point(u,v,d)
cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p;
    p.z = double(point.z) / camera.scale;
    p.x = ( point.x - camera.cx ) * p.z / camera.fx;
    p.y = ( point.y - camera.cy ) * p.z / camera.fy;

    return p;
}


//chaptr 4
void computeKeyPointAndDesp( FRAME& frame, string detector, string descriptor )
{
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;

    _detector = cv::FeatureDetector::create( detector.c_str() );
    _descriptor = cv::DescriptorExtractor::create( descriptor.c_str() );

    if(!_detector || !_descriptor)
    {
        cerr<<"Unknown detector or descriptor type ! "<<detector<<","<<descriptor<<endl;
        return;
    }
    assert(frame.rgb.data);
    _detector->detect( frame.rgb, frame.kp);
    _descriptor->compute( frame.rgb, frame.kp, frame.desp);

    return;
}


RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    RESULT_OF_PNP result;
    result.flag = 1;

    static ParameterReader pd;
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher;
    matcher.match( frame1.desp, frame2.desp, matches );

    cout<<"find total "<<matches.size()<<" matches."<<endl;
    vector<cv::DMatch> goodMatches;
    double minDis = 9999;
    double good_match_threshold = atof( pd.getData("good_match_threshold").c_str() );
    for(size_t i=0 ; i<matches.size() ; i++)
    {
        if(matches[i].distance < minDis)
            minDis = matches[i].distance;
    }

    cout<<"\nminDis = "<<minDis<<"\n"<<endl;

    if ( minDis < 10 )
        minDis = 10;
    for(size_t i=0 ; i<matches.size() ; i++)
    {
        if(matches[i].distance < good_match_threshold * minDis)
            goodMatches.push_back( matches[i] );
    }

//    cv::Mat imgGoodmatches;
//    cv::drawMatches(frame1.rgb, frame1.kp, frame2.rgb, frame2.kp, goodMatches, imgGoodmatches);
//    cv::imshow("good matches",imgGoodmatches);
//    cv::waitKey(666);
    cout<<"good matches : "<<goodMatches.size()<<endl;
    if( goodMatches.size() < 10 )
    {
        result.flag=0;
        return result;
    }

    vector<cv::Point3f> pts_obj;
    vector<cv::Point2f> pts_img;

    for(size_t i=0 ; i<goodMatches.size() ; i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d = frame1.depth.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d <= 0)
            continue;
        pts_img.push_back( cv::Point2f( frame2.kp[goodMatches[i].trainIdx].pt ) );

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, camera );
        pts_obj.push_back( pd );
    }
    if (pts_obj.size() ==0 || pts_img.size()==0)
    {
        result.inliers = -1;
        return result;
    }

    double camera_matrix_data[3][3] = {
            {camera.fx, 0, camera.cx},
            {0, camera.fy, camera.cy},
            {0,0,1}
    };
    cout<<"solving pnp ..."<<endl;

    cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers);


    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;

    cout<<"\ninliers : "<<inliers.rows<<endl;
    return result;
}


//chaptr 5
Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec )//matrix T
{
    cv::Mat R;
    cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    for ( int i=0; i<3; i++ )
        for ( int j=0; j<3; j++ )
            r(i,j) = R.at<double>(i,j);

    // 将平移向量和旋转矩阵转换成变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0,3) = tvec.at<double>(0,0);
    T(1,3) = tvec.at<double>(1,0);
    T(2,3) = tvec.at<double>(2,0);
    return T;
}


PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d& T, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    PointCloud::Ptr newCloud = image2PointCloud( newFrame.rgb, newFrame.depth, camera);

    PointCloud::Ptr output( new PointCloud() );
    pcl::transformPointCloud( *original, *output, T.matrix() );
    *newCloud += *output;

    //Voxel grid 滤波降采样
    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;
    double gridsize = atof( pd.getData("voxel_grid").c_str() );
    voxel.setLeafSize( gridsize, gridsize, gridsize);
    voxel.setInputCloud( newCloud );
    PointCloud::Ptr tmp( new PointCloud() );
    voxel.filter(*tmp);
    return tmp;
}

PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, PointCloud::Ptr newCloud, Eigen::Isometry3d& T, CAMERA_INTRINSIC_PARAMETERS& camera)
{
//    PointCloud::Ptr newCloud = image2PointCloud( newFrame.rgb, newFrame.depth, camera);

    PointCloud::Ptr output( new PointCloud() );
    pcl::transformPointCloud( *original, *output, T.matrix() );
    *newCloud += *output;
//    return newCloud;

    //Voxel grid 滤波降采样
    static pcl::VoxelGrid<PointT> voxel;
    double gridsize = 0.02 ;
    voxel.setLeafSize( gridsize, gridsize, gridsize);
    voxel.setInputCloud( newCloud );
    PointCloud::Ptr tmp( new PointCloud() );
    voxel.filter(*tmp);
    return tmp;
}



