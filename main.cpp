#include <iostream>
#include <opencv2/opencv.hpp>
#include "inference.h"
#include "TYImageProc.h"
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

int main() {
    Mat img = imread(R"(D:\files\data_0803\815\1\1_color.png)");
    Mat disData = imread(R"(D:\files\data_0803\815\1\1_depth.png)", IMREAD_UNCHANGED);
    TY_CAMERA_CALIB_INFO depth_calib;
    TY_CAMERA_CALIB_INFO color_calib;
    depth_calib.intrinsicWidth = 1280;
    depth_calib.intrinsicHeight = 960;
    depth_calib.intrinsic = {1042.5369673635596, 0.0, 662.4808235168457, 0.0, 1042.5369673635596,
                                    484.1883888244629, 0.0, 0.0, 1.0,};

    color_calib.intrinsicWidth = 2560;
    color_calib.intrinsicHeight = 1920;
    color_calib.intrinsic = {1853.9256227399726, 0.0, 1299.0400386885497, 0.0, 1853.3664688649599,
                                986.5979375266809, 0.0, 0.0, 1.0,};
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
        if (dect_i != 1)
            continue;
        Rect yolo_res = dect[dect_i];
        cv::Scalar color(255, 0, 0);
        cv::rectangle(show, yolo_res, color, 2);
        cv::Point2d roi_lefttop(yolo_res.x, yolo_res.y);
        cv::Point2d roi_rightbut(yolo_res.x + yolo_res.width, yolo_res.y + yolo_res.height);
        Mat mask, bin_img, roi, blurImage;
        mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
        cvtColor(img, gray_img, COLOR_BGR2GRAY);
        for (int i = roi_lefttop.x; i < roi_rightbut.x; i++) {
            for (int j = roi_lefttop.y; j < roi_rightbut.y; j++) {
                mask.at<uchar>(j, i) = 255; //at是行列走的，和xy坐标反过来
            }
        }
        cv::bitwise_and(gray_img, gray_img, roi, mask);
        imwrite(std::format("roi{}.png", dect_i), roi);
        cv::medianBlur(roi, blurImage, 7);
        imwrite(std::format("blur{}.png", dect_i), blurImage);
        std::vector<Vec3f> circles;
        cv::HoughCircles(blurImage, circles, cv::HOUGH_GRADIENT_ALT, 1.5, 100,
            300, 0.8, 5, 60);
        if (circles.size() != 2) {
            std::cout << "ERROR!" << std::endl;
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

        }
    }
    imwrite("dect.png", show);
    return 0;
}
