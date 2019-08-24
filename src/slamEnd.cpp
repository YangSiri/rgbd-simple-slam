//chap 6
#include<iostream>
#include<fstream>
#include<sstream>
using namespace std;

#include "slamBase.h"
#include<g2o/types/slam3d/types_slam3d.h>
#include<g2o/core/sparse_optimizer.h>
#include<g2o/core/block_solver.h>
#include<g2o/core/factory.h>
#include<g2o/core/optimization_algorithm_factory.h>
#include<g2o/core/optimization_algorithm_gauss_newton.h>
#include<g2o/core/robust_kernel.h>
#include<g2o/core/robust_kernel_factory.h>
#include<g2o/core/optimization_algorithm_levenberg.h>
#include<g2o/solvers/eigen/linear_solver_eigen.h>

FRAME readFrame (int index, ParameterReader& pd);

double normofTransform (cv::Mat rvec, cv::Mat tvec);

int main(int argc, char** argv)
{
    ParameterReader pd;
    int startIndex = atoi( pd.getData("start_index").c_str() );
    int endIndex = atoi( pd.getData("end_index").c_str() );

    cout<<"Initializing ..."<<endl;
    int currIndex = startIndex;
    FRAME lastFrame = readFrame( currIndex, pd );

    string detector = pd.getData("detector");
    string descriptor = pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();

    computeKeyPointAndDesp( lastFrame, detector, descriptor );
    PointCloud::Ptr cloud = image2PointCloud( lastFrame.rgb, lastFrame.depth, camera );

    pcl::visualization::CloudViewer viewer( "viewer" );
    bool visualize = pd.getData( "visualize_pointcloud" )==string("yes");

    int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    double max_norm = atof( pd.getData("max_norm").c_str() );

    //选择优化方法
    /*  typedef g2o::BlockSolver_6_3 SlamBlockSolver;
  typedef g2o::LinearSolverEigen< SlamBlockSolver::PoseMatrixType > SlamLinearSolver;

  SlamLinearSolver* linearSolver = new SlamLinearSolver();
  linearSolver->setBlockOrdering( false );
  SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );
*/

    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > SlamBlockSolver;
    SlamBlockSolver::LinearSolverType* linearSolver = new g2o::LinearSolverEigen <SlamBlockSolver::PoseMatrixType> ();
    SlamBlockSolver* solver_ptr = new SlamBlockSolver ( std::unique_ptr<SlamBlockSolver::LinearSolverType>(linearSolver) );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::unique_ptr<SlamBlockSolver>(solver_ptr));

    g2o::SparseOptimizer globalOptimizer;
    globalOptimizer.setAlgorithm( solver );
    globalOptimizer.setVerbose( false );//不要输出调试信息
    //向globalOptimizer增加第一个顶点
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId( currIndex );
    v->setEstimate( Eigen::Isometry3d::Identity() );//估计为单位矩阵
    v->setFixed( true );//第一个顶点固定，不用优化
    globalOptimizer.addVertex( v );

    int lastIndex = currIndex;
    Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();
    for( currIndex=startIndex+1 ; currIndex<endIndex ; currIndex++ )
    {
        cout<<"========================="<<endl;
        cout<<"Reading files "<<currIndex<<endl;
        FRAME currFrame = readFrame( currIndex, pd );
        computeKeyPointAndDesp( currFrame, detector, descriptor );
        //比较currFrame和lastFrame
        RESULT_OF_PNP result = estimateMotion( lastFrame, currFrame, camera );
        if( result.inliers < min_inliers || result.flag == 0)//inliers不够，放弃该帧
            continue;

        cout << "result.rvec " << result.rvec << endl;
        cout << "result.tvec " << result.tvec << endl;
        double norm = normofTransform(result.rvec, result.tvec);
        cout<<"norm = "<<norm<<"\n"<<endl;
        if( norm > max_norm )
            continue;

        // cloud = joinPointCloud( cloud, currFrame, T, camera );

        // 向g2o中增加这个顶点与上一帧联系的边
        // 顶点部分
        // 顶点只需设定id即可
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId( currIndex );
        v->setEstimate( Eigen::Isometry3d::Identity() );
        globalOptimizer.addVertex(v);
        g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct( "Cauchy" );
        // 边部分
        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        // 连接此边的两个顶点id
        edge->vertices()[0] = globalOptimizer.vertex( lastIndex );
        edge->vertices()[1] = globalOptimizer.vertex( currIndex );
        edge->setRobustKernel( robustKernel );
        // 信息矩阵
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();

        // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
        // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
        // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
        information(0,0) = information(1,1) = information(2,2) = 100;
        information(3,3) = information(4,4) = information(5,5) = 100;
        // 也可以将角度设大一些，表示对角度的估计更加准确
        edge->setInformation( information );

        //Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec );
        //T = T1.inverse() * T;
        //cout<<"T="<<T.matrix()<<endl;
        //cout<<"-----------------------------------"<<endl;
        //cout<<"T1="<<T1.matrix()<<endl;
        // 边的估计即是pnp求解之结果
        edge->setMeasurement( Eigen::Isometry3d::Identity() );//Segmentation fault (core dumped)!!!???

        // 将此边加入图中
        globalOptimizer.addEdge(edge);

        lastFrame = currFrame;
        lastIndex = currIndex;
        //T1=T;
    }

//    cout<<"=========================================="<<endl;
//    cout<<"Optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
//    globalOptimizer.save("./data/result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize( 100 );//可以指定优化步数
//    globalOptimizer.save("./data/result_after.g2o");
//    cout<<"Optimization done."<<endl;

    globalOptimizer.clear();
    return 0;
}


FRAME readFrame( int index, ParameterReader& pd )
{
    FRAME f;
    string rgbDir = pd.getData("rgb_dir");
    string depthDir = pd.getData("depth_dir");

    string rgbExt = pd.getData("rgb_extension");
    string depthExt = pd.getData("depth_extension");

    stringstream ss;
    ss<<rgbDir<<index<<rgbExt;
    string filename;
    ss>>filename;
    f.rgb = cv::imread(filename);

    ss.clear();
    filename.clear();
    ss<<depthDir<<index<<depthExt;
    ss>>filename;
    f.depth = cv::imread(filename,-1);
    f.frameID = index;
    return f;
}

double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
    return fabs( min( cv::norm(rvec), 2*M_PI-cv::norm(rvec) ) ) + fabs( cv::norm(tvec) );
}

