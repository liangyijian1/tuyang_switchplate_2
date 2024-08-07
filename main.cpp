#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "inference.h"
#include "TYImageProc.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

using namespace cv;

std::vector<Rect> dect_v8(const Mat& frame) {
    bool runOnGPU = false;
    Inference inf("./best.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);
    std::vector<Rect> res;
    std::vector<Detection> output = inf.runInference(frame);
    auto detections = output.size();
    std::cout << "Number of detections:" << detections << std::endl;
    for (int i = 0; i < detections; ++i)
    {
        Detection detection = output[i];
        cv::Rect box = detection.box;
        res.push_back(box);
    }
    return res;
}

void remove_3zero(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered) {
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        if (cloud->points[i].x == 0 && cloud->points[i].y == 0 && cloud->points[i].z == 0)
        {
            indices->indices.push_back(static_cast<int>(i));
        }
    }
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(indices);
    extract.setNegative(true);
    extract.filter(*cloud_filtered);
}

void visualize(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
    viewer.addCoordinateSystem(200.0);
    viewer.initCameraParameters();
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
        std::this_thread::sleep_for(std::chrono::microseconds(100000));
    }
}

void visualize(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("cloud show"));
    int v1 = 0;
    int v2 = 1;

    viewer->createViewPort(0, 0, 0.5, 1, v1);
    viewer->createViewPort(0.5, 0, 1, 1, v2);
    viewer->setBackgroundColor(0, 0, 0, v1);
    viewer->setBackgroundColor(0, 0, 0, v2);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cloud1, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> after_sac(cloud2, 0, 0, 255);

    viewer->addPointCloud(cloud1, color, "cloud", v1);
    viewer->addPointCloud(cloud2, after_sac, "plane cloud", v2);


    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::microseconds(10000));
    }
}

int main() {
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);
    Mat img = imread(R"(D:\files\data_0803\815\1\1_color.png)");
    Mat disData = imread(R"(D:\files\data_0803\815\1\1_depth.png)", IMREAD_UNCHANGED);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (R"(D:\files\data_0803\815\1\1.pcd)", *cloud) == -1)
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

    TY_CAMERA_CALIB_INFO depth_calib;
    TY_CAMERA_CALIB_INFO color_calib;
    depth_calib.intrinsicWidth = 1280;
    depth_calib.intrinsicHeight = 960;
    depth_calib.intrinsic = {1042.5369673635596, 0.0, 662.4808235168457, 0.0, 1042.5369673635596,
                                    484.1883888244629, 0.0, 0.0, 1.0,};

    color_calib.intrinsicWidth = 2560;
    color_calib.intrinsicHeight = 1920;
    color_calib.intrinsic = {1853.9256227399726, 0.0, 1299.0400386885497,
                                0.0, 1853.3664688649599, 986.5979375266809,
                                0.0, 0.0, 1.0,};
    color_calib.extrinsic = {0.9999728647705215, 0.005016437867965373, 0.005394911839227006, 25.073762805600555,
                                -0.004980224708919019, 0.9999651190176755, -0.006705080895485358, -0.05827761134692729,
                                -0.005428359281114419, 0.006678031078332451, 0.9999629677225212, -0.0843239650174262,
                                0.0, 0.0, 0.0, 1.0};
    color_calib.distortion = {-0.2297000739763372, 0.5169755815201543, 0.00044487206439792836, 0.0010329685399318549,
                                0.32840785996432853, 0.03494625623772451, 0.34850142249482163, 0.5474387187338604,
                                -0.0020897526645500017, 0.00032269378472499005, -0.0004825373377142142, 0.00013667696455172554};
    Mat map_disData = cv::Mat::zeros(img.size(), CV_16U);
    TYMapDepthImageToColorCoordinate(&depth_calib, disData.cols, disData.rows, disData.ptr<uint16_t>(),
                                    &color_calib, map_disData.cols, map_disData.rows, map_disData.ptr<uint16_t>());
    imwrite("map_disData.png", map_disData);

    Mat show, gray_img;
    img.copyTo(show);
    std::vector<Rect> dect = dect_v8(img);
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    for (int dect_i = 0; dect_i < dect.size(); dect_i++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_passthrough(new pcl::PointCloud<pcl::PointXYZ>);
        Rect yolo_res = dect[dect_i];
        cv::Scalar color(255, 0, 0);
        cv::rectangle(show, yolo_res, color, 2);
        cv::Point2d roi_lefttop(yolo_res.x, yolo_res.y);
        cv::Point2d roi_rightbut(yolo_res.x + yolo_res.width, yolo_res.y + yolo_res.height);
        Mat mask, bin_img, roi, blurImage;
        mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
        cvtColor(img, gray_img, COLOR_BGR2GRAY);
        for (int i = static_cast<int>(roi_lefttop.x); i < roi_rightbut.x; i++) {
            for (int j = static_cast<int>(roi_lefttop.y); j < roi_rightbut.y; j++) {
                mask.at<uchar>(j, i) = 255; //at是行列走的，和xy坐标反过来
            }
        }
        cv::bitwise_and(gray_img, gray_img, roi, mask);
        imwrite(std::format("roi{}.png", dect_i), roi);
        cv::medianBlur(roi, blurImage, 7);
        imwrite(std::format("blur{}.png", dect_i), blurImage);
        std::vector<Vec3f> circles;
        cv::HoughCircles(blurImage, circles, cv::HOUGH_GRADIENT_ALT, 1.5, 100,
            300, 0.8, 20, 60);
        if (circles.size() != 2) {
            std::cout << "ERROR!" << std::endl;
            Mat show_error;
            img.copyTo(show_error);
            for (auto &circle : circles) {
                //可视化圆心
                int radius = cvRound(circle[2]);
                Point center(cvRound(circle[0]), cvRound(circle[1]));
                cv::circle(show_error, center, 1, Scalar(0, 255, 0), -1, 1, 0);
                cv::circle(show_error, center, radius, Scalar(0, 0, 255), 2, 1, 0);
            }
            imwrite("show_error.png", show_error);
        }
        else {
            std::vector<std::vector<std::vector<std::vector<int>>>> circle_neigh_set;
            for (auto & circle : circles) {
                std::vector<std::vector<std::vector<int>>> neigh;
                std::vector<std::vector<int>> tmp_point(3, std::vector<int>(1, 0));
                std::vector<std::vector<int>> bias_vec(4, std::vector<int>(2, 0));
                int radius = cvRound(circle[2]);
                int radius_bias = radius + 10;
                bias_vec[0][0] = radius_bias;
                bias_vec[0][1] = 0;
                bias_vec[1][0] = 0;
                bias_vec[1][1] = -radius_bias;
                bias_vec[2][0] = -radius_bias;
                bias_vec[2][1] = 0;
                bias_vec[3][0] = 0;
                bias_vec[3][1] = radius_bias;
                for (size_t j = 0; j < 4; j++) {
                    int u = cvRound(circle[0]) + bias_vec[j][0];
                    int v = cvRound(circle[1]) + bias_vec[j][1];
                    auto Z = map_disData.at<unsigned short>(v, u);
                    tmp_point[0][0] = u;
                    tmp_point[1][0] = v;
                    tmp_point[2][0] = Z;

                    neigh.push_back(tmp_point);
                }
                circle_neigh_set.push_back(neigh);
                //可视化圆心
                Point center(cvRound(circle[0]), cvRound(circle[1]));
                cv::circle(show, center, 1, Scalar(0, 255, 0), -1, 1, 0);
                cv::circle(show, center, radius, Scalar(0, 0, 255), 1, 1, 0);
            }
            //寻找最高点
            int min_Z = 9999;
            int idx_min_Z_circle = -1;
            for (size_t i = 0; i < circle_neigh_set.size(); i++) {
                //遍历neigh
                for (size_t j = 0; j < 4; j++) {
                    int Z = circle_neigh_set[i][j][2][0];
                    if (Z == 0) {
                        break;
                    }
                    if (Z < min_Z) {
                        min_Z = Z;
                        idx_min_Z_circle = i;
                    }
                }
            }
            //可视化最高圆
            Point p(circles[idx_min_Z_circle][0], circles[idx_min_Z_circle][1]);
            cv::circle(show, p, 20, Scalar(255, 255, 255), -1, 1, 0);
            //配对圆
            int pair_circle = 0;
            if (pair_circle == idx_min_Z_circle) {
                pair_circle = abs(idx_min_Z_circle - 1);
            }

            //将YOLO结果转成XYZ形式，对点云进行裁剪
            auto roi_lefttop_Z = map_disData.at<uint16_t>(static_cast<int>(roi_lefttop.y), static_cast<int>(roi_lefttop.x));
            auto roi_lefttop_X = (roi_lefttop.x - color_calib.intrinsic.data[2]) * roi_lefttop_Z / color_calib.intrinsic.data[0];
            auto roi_lefttop_Y = (roi_lefttop.y - color_calib.intrinsic.data[5]) * roi_lefttop_Z / color_calib.intrinsic.data[4];

            auto roi_rightbut_Z = map_disData.at<uint16_t>(static_cast<int>(roi_rightbut.y), static_cast<int>(roi_rightbut.x));
            auto roi_rightbut_X = (roi_rightbut.x - color_calib.intrinsic.data[2]) * roi_rightbut_Z / color_calib.intrinsic.data[0];
            auto roi_rightbut_Y = (roi_rightbut.y - color_calib.intrinsic.data[5]) * roi_rightbut_Z / color_calib.intrinsic.data[4];

            //直通滤波
            pcl::PassThrough<pcl::PointXYZ> pass;
            pass.setInputCloud(cloud);
            pass.setFilterFieldName("z");
            pass.setFilterLimits(1220, 5000);
            pass.setNegative(true);
            pass.filter(*cloud_passthrough);

            pass.setInputCloud(cloud_passthrough);
            pass.setFilterFieldName("x");
            pass.setFilterLimits(roi_lefttop_X, roi_rightbut_X);
            pass.setNegative(false);
            pass.filter(*cloud_passthrough);

            pass.setInputCloud(cloud_passthrough);
            pass.setFilterFieldName("y");
            pass.setFilterLimits(roi_lefttop_Y, roi_rightbut_Y);
            pass.setNegative(false);
            pass.filter(*cloud_passthrough);

            //平面拟合
            pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_plane(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_passthrough));
            pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_plane);// 定义RANSAC算法模型
            ransac.setDistanceThreshold(5);// 设定距离阈值
            ransac.setMaxIterations(1000);     // 设置最大迭代次数
            ransac.setProbability(0.99);      // 设置从离群值中选择至少一个样本的期望概率
            ransac.computeModel();            // 拟合平面
            std::vector<int> inliers;              // 用于存放内点索引的vector
            ransac.getInliers(inliers);       // 获取内点索引
            Eigen::VectorXf coeff;
            ransac.getModelCoefficients(coeff);  //获取拟合平面参数，coeff分别按顺序保存a,b,c,d
            cout << "平面模型系数coeff(a,b,c,d): " << coeff[0] << " \t" << coeff[1] << "\t " << coeff[2] << "\t " << coeff[3] << endl;
            visualize(cloud, cloud_passthrough);
        } //else

    }
    imwrite("dect.png", show);
    return 0;
}
