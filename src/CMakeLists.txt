

FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters surface keypoints )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

FIND_PACKAGE( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

ADD_DEFINITIONS( ${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS} )
LINK_LIBRARIES( ${PCL_LIBRARIES_DIRS} )


ADD_LIBRARY( slamBase slamBase.cpp )
TARGET_LINK_LIBRARIES ( slamBase ${OpenCV_LIBS} ${PCL_LIBRARIES} )

LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( TBB_LIBS "/usr/lib/x86_64-linux-gnu/libtbb.so.2" )

ADD_LIBRARY( pca_analysis pca_analysis.cpp )
TARGET_LINK_LIBRARIES ( pca_analysis ${OpenCV_LIBS} ${PCL_LIBRARIES} ${TBB_LIBS})

ADD_LIBRARY( keypointExtraction keypointExtraction.cpp )
TARGET_LINK_LIBRARIES ( keypointExtraction pca_analysis ${OpenCV_LIBS} ${PCL_LIBRARIES} )

ADD_LIBRARY( descriptor_geo_struc_color descriptor_geo_struc_color.cpp )
TARGET_LINK_LIBRARIES ( descriptor_geo_struc_color  pca_analysis ${OpenCV_LIBS} ${PCL_LIBRARIES} )

# add executables
ADD_EXECUTABLE( generate_pointcloud generatePointCloud.cpp )
TARGET_LINK_LIBRARIES( generate_pointcloud ${OpenCV_LIBS} ${PCL_LIBRARIES} )

ADD_EXECUTABLE( detectFeatures detectFeatures.cpp )
TARGET_LINK_LIBRARIES( detectFeatures slamBase ${OpenCV_LIBS} ${PCL_LIBRARIES} )

ADD_EXECUTABLE( joinPointCloud joinPointCloud.cpp )
TARGET_LINK_LIBRARIES( joinPointCloud slamBase ${OpenCV_LIBS} ${PCL_LIBRARIES} )

ADD_EXECUTABLE( visualOdometry visualOdometry.cpp )
TARGET_LINK_LIBRARIES( visualOdometry slamBase ${OpenCV_LIBS} ${PCL_LIBRARIES} )

# 添加g2o的依赖
# 因为g2o不是常用库，要添加它的findg2o.cmake文件
LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O REQUIRED )

# CSparse
FIND_PACKAGE( CSparse REQUIRED )
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )

ADD_EXECUTABLE( slamEnd slamEnd.cpp )
TARGET_LINK_LIBRARIES( slamEnd
    slamBase
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})

MESSAGE(INFO ${PCL_LIBRARIES})

#chap 7
ADD_EXECUTABLE( slam slam.cpp )
TARGET_LINK_LIBRARIES( slam
    slamBase
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})

#feature training DBoW3
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.a" )

add_executable( feature_training feature_training.cpp )
target_link_libraries( feature_training ${OpenCV_LIBS} ${DBoW3_LIBS} ${PCL_LIBRARIES})

add_executable(image_matching image_matching.cpp)
target_link_libraries(image_matching slamBase ${OpenCV_LIBS} ${DBoW3_LIBS} ${PCL_LIBRARIES})

add_executable(spatial_feature spatial_feature.cpp)
target_link_libraries(spatial_feature slamBase keypointExtraction descriptor_geo_struc_color ${OpenCV_LIBS} ${DBoW3_LIBS} ${PCL_LIBRARIES})