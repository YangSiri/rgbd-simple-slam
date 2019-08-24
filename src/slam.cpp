
//chap 7
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "slamBase.h"

#include<pcl/filters/voxel_grid.h>
#include<pcl/filters/passthrough.h>


#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > SlamBlockSolver;

FRAME readFrame( int index, ParameterReader& pd);

double normofTransform( cv::Mat rvec, cv::Mat tvec );

enum CHECK_RESULT {NOT_MATCHED=0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME };

CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops=false );

void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti );

void checkRandomLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti );

int main( int argc, char** argv )
{
    ParameterReader pd;
    int startIndex = atoi( pd.getData("start_index").c_str() );
    int endIndex = atoi( pd.getData("end_index").c_str() );

    vector<FRAME> keyframes;

    cout<<"Initializing ..."<<endl;
    int currIndex = startIndex;
    FRAME currFrame = readFrame( currIndex, pd );

    string detector = pd.getData("detector");
    string descriptor = pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();

    computeKeyPointAndDesp( currFrame, detector, descriptor );
    PointCloud::Ptr cloud = image2PointCloud( currFrame.rgb, currFrame.depth, camera );

    //图优化部分
    SlamBlockSolver::LinearSolverType* linearSolver = new g2o::LinearSolverEigen <SlamBlockSolver::PoseMatrixType> ();
    SlamBlockSolver* solver_ptr = new SlamBlockSolver ( std::unique_ptr<SlamBlockSolver::LinearSolverType>(linearSolver) );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::unique_ptr<SlamBlockSolver>(solver_ptr));

    g2o::SparseOptimizer globalOptimizer;
    globalOptimizer.setAlgorithm( solver );
    globalOptimizer.setVerbose( false );

    //startIndex为Pose graph中第一个vertex
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId( currIndex );
    v->setEstimate( Eigen::Isometry3d::Identity() );
    v->setFixed (true);
    globalOptimizer.addVertex( v ) ;

    keyframes.push_back( currFrame );

    double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    bool check_loop_closure = pd.getData( "check_loop_closure" )==string("yes") ;

    int count=0;//keyframes.size()
    for( currIndex = startIndex+1 ; currIndex<endIndex; currIndex++)
    {
        cout<<"===================="<<endl;
        cout<<"Reading files "<<currIndex<<endl;
        FRAME currFrame = readFrame( currIndex, pd );
        computeKeyPointAndDesp( currFrame, detector, descriptor );
        CHECK_RESULT result = checkKeyframes( keyframes.back(), currFrame, globalOptimizer );
        switch(result)
        {
            case NOT_MATCHED:
                cout<<"Not enough inliers."<<endl;
                break;
            case TOO_FAR_AWAY:
                cout<<"Too far away, may be an error."<<endl;
                break;
            case TOO_CLOSE:
                cout<<"Too close, not a keyframe"<<endl;
                break;
            case KEYFRAME:
                cout<<"This is a new keyframe"<<endl;
                count++;
//                if( check_loop_closure )
//                {
//                    checkNearbyLoops( keyframes, currFrame, globalOptimizer );
//                    checkRandomLoops( keyframes, currFrame, globalOptimizer );
//                }
//                keyframes.push_back( currFrame );
                break;
            default :
                break;
        }
    }
    cout<<"\n keyframes : "<<count<<endl;
//    cout<<"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
//    globalOptimizer.save( "./result_before2.g2o" );
//    globalOptimizer.initializeOptimization();
//    globalOptimizer.optimize( 100 );
//    globalOptimizer.save( "./result_after2.g2o" );
//    cout<<"Optimization done."<<endl;
//
//    cout<<"saving the point cloud map ..."<<endl;
//    PointCloud::Ptr output ( new PointCloud() );
//    PointCloud::Ptr tmp ( new PointCloud() );
//
//    pcl::VoxelGrid<PointT> voxel; // 网格滤波器，调整地图分辨率
//    pcl::PassThrough<PointT> pass; // z方向区间滤波器，由于rgbd相机的有效深度区间有限，把太远的去掉
//    pass.setFilterFieldName("z");
//    pass.setFilterLimits( 0.0, 4.0 ); //4m以上就不要了
//
//    double gridsize = atof( pd.getData("voxel_grid").c_str() );
//    voxel.setLeafSize( gridsize, gridsize, gridsize );
//
//    for (size_t i=0; i<keyframes.size(); i++)
//    {
//        // 从g2o里取出一帧
//        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[i].frameID ));
//        Eigen::Isometry3d pose = vertex->estimate(); //该帧优化后的位姿
//        PointCloud::Ptr newCloud = image2PointCloud( keyframes[i].rgb, keyframes[i].depth, camera ); //转成点云
//        // 以下是滤波
//        voxel.setInputCloud( newCloud );
//        voxel.filter( *tmp );
//        pass.setInputCloud( tmp );
//        pass.filter( *newCloud );
//        // 把点云变换后加入全局地图中
//        pcl::transformPointCloud( *newCloud, *tmp, pose.matrix() );
//        *output += *tmp;
//        tmp->clear();
//        newCloud->clear();
//    }
//
//    voxel.setInputCloud( output );
//    voxel.filter( *tmp );
//    pcl::io::savePCDFile( "./result_of_slam.pcd", *tmp );
//    cout<<"Final map is saved."<<endl;
    return 0;
}


FRAME readFrame( int index, ParameterReader& pd )
{
    FRAME f;
    string rgbDir   =   pd.getData("rgb_dir");
    string depthDir =   pd.getData("depth_dir");

    string rgbExt   =   pd.getData("rgb_extension");
    string depthExt =   pd.getData("depth_extension");

    stringstream ss;
    ss<<rgbDir<<index<<rgbExt;
    string filename;
    ss>>filename;
    f.rgb = cv::imread( filename );

    ss.clear();
    filename.clear();
    ss<<depthDir<<index<<depthExt;
    ss>>filename;

    f.depth = cv::imread( filename, -1 );
    f.frameID = index;
    return f;
}


double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}


CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops)
{
    static ParameterReader pd;
    static int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    static double max_norm = atof( pd.getData("max_norm").c_str() );
    static double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    static double max_norm_lp = atof( pd.getData("max_norm_lp").c_str() );
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();

    RESULT_OF_PNP result = estimateMotion( f1, f2, camera );
    if(result.inliers < min_inliers || result.flag == 0)
        return NOT_MATCHED;
    //计算运动范围
    double norm = normofTransform(result.rvec, result.tvec);
    cout<<"norm = "<<norm<<endl;

    if (norm >= (is_loops == false ? max_norm : max_norm_lp))
        return TOO_FAR_AWAY;

    if ( norm <= keyframe_threshold )
        return TOO_CLOSE;   // too adjacent frame
    // 向g2o中增加这个顶点与上一帧联系的边
    // 顶点部分
    // 顶点只需设定id即可
//    if (is_loops == false)
//    {
//        g2o::VertexSE3 *v = new g2o::VertexSE3();
//        v->setId( f2.frameID );
//        v->setEstimate( Eigen::Isometry3d::Identity() );
//        opti.addVertex(v);
//    }
//    // 边部分
//    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
//    // 连接此边的两个顶点id
//    edge->setVertex( 0, opti.vertex(f1.frameID ));
//    edge->setVertex( 1, opti.vertex(f2.frameID ));
//    edge->setRobustKernel( new g2o::RobustKernelHuber() );
//    // 信息矩阵
//    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
//    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
//    // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
//    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
//    information(0,0) = information(1,1) = information(2,2) = 100;
//    information(3,3) = information(4,4) = information(5,5) = 100;
//    edge->setInformation( information );
//
//    Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec );
//    edge->setMeasurement( T );
//    opti.addEdge( edge );
    return KEYFRAME;
}

void checkRandomLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti)
{
    static ParameterReader pd;
    static int random_loops = atoi( pd.getData("random_loops").c_str() );
    srand( (unsigned int)time (NULL));

    if(frames.size() <= random_loops)
    {
        cout<<"no enough keyframes, check everyone"<<endl;
        for(size_t i=0 ; i<frames.size() ; i++)
            checkKeyframes(frames[i], currFrame, opti, true);
    }
    else
    {
        cout<<"randomly check loops"<<endl;
        for(int i=0 ; i<random_loops ; i++)
        {
            int index = rand()%frames.size();
            checkKeyframes(frames[index], currFrame, opti, true);
        }
    }

}


void checkNearbyLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti)
{
    static ParameterReader pd;
    static int nearby_loops = atoi( pd.getData("nearby_loops").c_str() );

    if( frames.size() <= nearby_loops )
    {
        cout<<"no enough frames, check everyone"<<endl;
        for(size_t i=0 ; i<frames.size() ; i++)
            checkKeyframes(frames[i], currFrame, opti, true);
    }
    else
    {
        cout<<"check the nearest ones"<<endl;
        for(size_t i=frames.size()-nearby_loops ; i<frames.size() ; i++)
            checkKeyframes(frames[i], currFrame, opti, true);
    }
}
