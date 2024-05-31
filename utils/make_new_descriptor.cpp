#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <filesystem>
#include <string>
#include <optional>
#include <tuple>
#include <iomanip>
#include <pcl/io/vtk_lib_io.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
// #include <tf/transform_datatypes.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/search/impl/search.hpp>
// #include <pcl/range_image/range_image.h>
// #include <pcl/common/common.h>
// #include <pcl/common/transforms.h>
// #include <pcl/filters/extract_indices.h>
// #include <pcl/registration/icp.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/keypoints/harris_3d.h>
// #include <pcl/features/organized_edge_detection.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/PCLPointCloud2.h>
// #include <pcl/console/parse.h>
// #include <pcl/console/print.h>
// #include <pcl/console/time.h>
// #include <pcl/features/integral_image_normal.h>

// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/octree/octree_pointcloud_voxelcentroid.h>
// #include <pcl/filters/crop_box.h> 
// #include <pcl/filters/statistical_outlier_removal.h>
// #include <pcl/filters/passthrough.h>
// #include <pcl/segmentation/sac_segmentation.h>
// #include <pcl/segmentation/extract_clusters.h>
// #include <pcl/features/normal_3d_omp.h>

// #include <pcl/point_types_conversion.h>
// #include <pcl/surface/gp3.h>
// #include <pcl/features/rops_estimation.h>

#include <Eigen/Dense>

// #include <ceres/ceres.h>

namespace fs = std::filesystem;

struct PointCloudData {
    Eigen::MatrixXd points;   // 좌표 및 확장된 1열
    Eigen::VectorXd scores;   // 점수
    Eigen::MatrixXd descriptors;  // FPFH descriptor

    // 생성자: 초기 데이터 크기 설정
    PointCloudData(int rows) : points(rows, 4), scores(rows), descriptors(rows, 33) {}
};

PointCloudData readAndProcessBinaryFile(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    std::vector<double> buffer;

    if (!file) {
        std::cerr << "Cannot open the file!" << std::endl;
        throw std::runtime_error("File cannot be opened.");
    }

    // 파일의 전체 내용을 읽어 버퍼에 저장
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    buffer.resize(size / sizeof(float));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();

    // 데이터 행렬로 변환
    
    int rows = buffer.size() / 37;
    Eigen::Map<Eigen::MatrixXd> data(buffer.data(), rows, 37);

    cout << "rows size: " << rows << endl;

    // 결과 데이터 구조체 초기화
    PointCloudData pcData(rows);

    cout << data << endl;

    // 좌표 데이터 및 1열 추가
    pcData.points.leftCols(3) = data.leftCols(3);
    pcData.points.col(3) = Eigen::VectorXd::Ones(rows);

    // 점수
    pcData.scores = data.col(3);

    // 디스크립터
    pcData.descriptors = data.rightCols(33);

    return pcData;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr readPointCloudFromFile(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "파일을 열 수 없습니다: " << file_path << std::endl;
        return nullptr;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    while (!file.eof()) {
        pcl::PointXYZI point;
        file.read(reinterpret_cast<char*>(&point.x), sizeof(float));
        file.read(reinterpret_cast<char*>(&point.y), sizeof(float));
        file.read(reinterpret_cast<char*>(&point.z), sizeof(float));
        file.read(reinterpret_cast<char*>(&point.intensity), sizeof(float));

        cloud->push_back(point);
    }

    file.close();

    return cloud;
}

int main(int argc, char **argv) {
    std::string base_path = "/media/vision/Seagate/DataSets/kitti/dataset/sequences";
    std::string base_keypoints_path = "/home/vision/ADD_prj/MDGAT-matcher/KITTI/keypoints/tsf_256_FPFH_16384-512-k1k16-2d-nonoise";

    // 시퀸스 0부터 10까지 순회
    for (int seq = 0; seq <= 0; ++seq) {
        std::stringstream ss;
        ss << std::setw(2) << std::setfill('0') << seq;
        std::string sequence = ss.str();
        std::string velodyne_path = base_path + "/" + sequence + "/velodyne/";
        std::string keypoints_path = base_keypoints_path + "/" + sequence + "/";

        // 해당 디렉토리의 모든 .bin 파일 순회
        for (const auto& entry : fs::directory_iterator(velodyne_path)) {
            if (entry.path().extension() == ".bin") {
                // pointcloud 파일 읽기
                std::string cloud_file_path = entry.path().string();
                pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
                cloud = readPointCloudFromFile(cloud_file_path);

                // keypoints 파일 읽기
                std::string keypoints_file_path = keypoints_path + entry.path().filename().string();
                PointCloudData pcData = readAndProcessBinaryFile(keypoints_file_path);

                // int i = 0;
                // 결과 출력
                // std::cout << "Points and Ones:" << std::endl;
                // std::cout << pcData.points << std::endl;
                // std::cout << "Scores:" << std::endl;
                // std::cout << pcData.scores << std::endl;
                // std::cout << "Descriptors:" << std::endl;
                // std::cout << pcData.descriptors << std::endl;

                // cout << "keypoints num: " << i << endl;


                // pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
                // viewer.setBackgroundColor(0, 0, 0); // 배경 색 설정: 검은색
                // viewer.addPointCloud<pcl::PointXYZI>(cloud, "sample cloud");
                // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
                // viewer.spin();

                std::cout << "파일 처리: " << cloud_file_path << ", " << keypoints_file_path << std::endl;

                break;
            }
        }

        

    }

    return 0;
}
