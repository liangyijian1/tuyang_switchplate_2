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
#include <pcl/common/centroid.h>
#include <boost/thread/thread.hpp>
#include <opencv2/core/eigen.hpp>

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


bool isRotatedMatrix(Mat& R)        //旋转矩阵的转置矩阵是它的逆矩阵，逆矩阵 * 矩阵 = 单位矩阵
{
    Mat temp33 = R({
    0,0,3,3 }); //无论输入是几阶矩阵，均提取它的三阶矩阵
    Mat Rt;
    transpose(temp33, Rt);  //转置矩阵
    Mat shouldBeIdentity = Rt * temp33;//是旋转矩阵则乘积为单位矩阵
    Mat I = Mat::eye(3, 3, shouldBeIdentity.type());

    return cv::norm(I, shouldBeIdentity) < 1e-6;
}

Mat eulerAngleToRotateMatrix(const Mat& eulerAngle, const std::string& seq)
{

    CV_Assert(eulerAngle.rows == 1 && eulerAngle.cols == 3);//检查参数是否正确

    eulerAngle /= (180 / CV_PI);        //度转弧度

    Matx13d m(eulerAngle);              //<double, 1, 3>

    auto rx = m(0, 0), ry = m(0, 1), rz = m(0, 2);
    auto rxs = sin(rx), rxc = cos(rx);
    auto rys = sin(ry), ryc = cos(ry);
    auto rzs = sin(rz), rzc = cos(rz);

    //XYZ方向的旋转矩阵
    Mat RotX = (Mat_<double>(3, 3) <<
        1, 0, 0,
        0, rxc, -rxs,
        0, rxs, rxc);
    Mat RotY = (Mat_<double>(3, 3) <<
        ryc, 0, rys,
        0,    1, 0,
        -rys, 0, ryc);
    Mat RotZ = (Mat_<double>(3, 3) <<
        rzc, -rzs, 0,
        rzs, rzc, 0,
        0, 0, 1);
    //按顺序合成后的旋转矩阵
    cv::Mat rotMat;
    if (seq == "zyx") rotMat = RotX * RotY * RotZ;
    else if (seq == "yzx") rotMat = RotX * RotZ * RotY;
    else if (seq == "zxy") rotMat = RotY * RotX * RotZ;
    else if (seq == "yxz") rotMat = RotZ * RotX * RotY;
    else if (seq == "xyz") rotMat = RotZ * RotY * RotX;
    else if (seq == "xzy") rotMat = RotY * RotZ * RotX;
    else
    {
        cout << "Euler Angle Sequence string is wrong...";
    }
    if (!isRotatedMatrix(rotMat))       //欧拉角特殊情况下会出现死锁
    {
        cout << "Euler Angle convert to RotatedMatrix failed..." << endl;
        exit(-1);
    }
    return rotMat;
}

Mat attitudeVectorToMatrix(const Mat& m, const std::string& seq)
{

    CV_Assert(m.total() == 6);
    Mat temp = Mat::eye(4, 4, CV_64FC1);
    Mat rotVec;
    if (m.total() == 6)
    {
        rotVec = m({3,0,3,1 });   //读取存储的欧拉角
    }
    //如果seq为空，表示传入的是3*1旋转向量，否则，传入的是欧拉角
    if (0 == seq.compare(""))
    {
        Rodrigues(rotVec, temp({0,0,3,3 }));   //罗德利斯转换
    }
    else
    {
        eulerAngleToRotateMatrix(rotVec, seq).copyTo(temp({0,0,3,3 }));
    }
    //存入平移矩阵
    temp({
    3,0,1,3 }) = m({
    0,0,3,1 }).t() * 1000;
    return temp;   //返回转换结束的齐次矩阵
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
    Mat_<double> calib_pose = (Mat_<double>(1, 6) <<
        0.0840643, 0.168102, -0.201074, -3.13157, 3.13194, 0.00268412
    );


    Mat hand_eye_mat = attitudeVectorToMatrix(calib_pose.row(0), "xyz");
    cout << "calib_mat: \n" << hand_eye_mat << endl;
    Eigen::Isometry3d hand_eye_eigen = Eigen::Isometry3d::Identity();
    cv::cv2eigen(hand_eye_mat, hand_eye_eigen.matrix());
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
                int radius_bias = radius + 15;
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
                        continue;
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
            int neigh_max = -1;
            int neigh_max_idx = -1;
            for (size_t i = 0; i < circle_neigh_set[idx_min_Z_circle].size(); i++) {
                auto tmp = circle_neigh_set[idx_min_Z_circle][i][2][0];
                if (tmp == 0) {
                    neigh_max_idx = i;
                    break;
                }
                if (tmp > neigh_max) {
                    neigh_max = tmp;
                    neigh_max_idx = i;
                }
            }
            circle_neigh_set[idx_min_Z_circle].erase(circle_neigh_set[idx_min_Z_circle].begin() + neigh_max_idx);
            for (auto & circle_neigh : circle_neigh_set[idx_min_Z_circle]) {
                cv::circle(show, Point(circle_neigh[0][0], circle_neigh[1][0]), 1, Scalar(255, 255, 0), -1, 1, 0);
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
            pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_plane);
            ransac.setDistanceThreshold(5);
            ransac.setMaxIterations(1000);
            ransac.setProbability(0.99);
            ransac.computeModel();
            std::vector<int> inliers;
            ransac.getInliers(inliers);
            Eigen::VectorXf coeff;
            ransac.getModelCoefficients(coeff);
            cout << "plane coeff(a,b,c,d): " << coeff[0] << " \t" << coeff[1] << "\t " << coeff[2] << "\t " << coeff[3] << endl;
            Eigen::Vector4f centroid;					// TODO: 可以使用内点去求中心，提高中心定位精度；该方法要确保点云完整；
            pcl::compute3DCentroid(*cloud_passthrough, centroid);
            auto center_point = centroid.head<3>().transpose();
            cout << "center_point res：" << center_point << endl;

            //计算平面法向量
            Eigen::Vector3d norm;
            norm[0] = coeff[0];
            norm[1] = coeff[1];
            norm[2] = coeff[2];
            if (norm[2] < 0) {
                norm = -1 * norm;
                cout << std::format("norm_ROT_0, norm_ROT_1, norm_ROT_2:{:.3f}, {:.3f}, {:.3f}\n", norm[0], norm[1], norm[2]);
            }

            Eigen::Vector3d v1;
            float c1Z = 0;
            for (auto & neigh_point : circle_neigh_set[idx_min_Z_circle]) {
                c1Z = c1Z + neigh_point[2][0];
            }
            c1Z = c1Z / circle_neigh_set[idx_min_Z_circle].size();
            auto c1X = (circles[idx_min_Z_circle][0] - color_calib.intrinsic.data[2]) * c1Z / color_calib.intrinsic.data[0];
            auto c1Y = (circles[idx_min_Z_circle][1] - color_calib.intrinsic.data[5]) * c1Z / color_calib.intrinsic.data[4];
            cout << std::format("c1 XYZ is : {}, {}, {}", c1X, c1Y, c1Z) << endl;
            //v1为Y法向量，并固定v1向量方向
            if (c1X < center_point[0]) {
                v1[0] = c1X - center_point[0];
                v1[1] = c1Y - center_point[1];
                v1[2] = c1Z - center_point[2];
            }
            else {
                cout << "Flip the v1 direction\n";
                v1[0] = center_point[0] - c1X;
                v1[1] = center_point[1] - c1Y;
                v1[2] = center_point[2] - c1Z;
            }
            //v2为X向量
            Eigen::Vector3d v2 = norm.cross(v1);
            //单位化三个向量
            norm.normalize();
            v1.normalize();
            v2.normalize();

            //垫板->相机
            Eigen::Matrix3d R_t_c = Eigen::Matrix3d::Identity();
            R_t_c <<
                v2[0], v1[0], norm[0],
                v2[1], v1[1], norm[1],
                v2[2], v1[2], norm[2];
            Eigen::Vector3d T_t_c;
            T_t_c << center_point[0], center_point[1], center_point[2];

            Eigen::Isometry3d RT_t_c = Eigen::Isometry3d::Identity();
            RT_t_c.rotate(R_t_c);
            RT_t_c.pretranslate(T_t_c);
            cout << "target->camera:\n" << RT_t_c.matrix() << endl;
            Eigen::Vector3d eulerAngle = RT_t_c.rotation().eulerAngles(2, 1, 0);
            eulerAngle[0] = eulerAngle[0] * 180 / M_PI;
            eulerAngle[1] = eulerAngle[1] * 180 / M_PI;
            eulerAngle[2] = eulerAngle[2] * 180 / M_PI;
            cout << "eulerAngle:\n" << eulerAngle << endl;
            cout << "**************************************\n" << endl;

            auto RT_t_g = hand_eye_eigen * RT_t_c;
            cout << "RT_t_g:\n" << RT_t_g.matrix() << endl;

            //visualize(cloud, cloud_passthrough);
        } //else
    }
    imwrite("dect.png", show);
    return 0;
}
