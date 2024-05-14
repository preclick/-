#include<opencv2/opencv.hpp>
#include"quickopencv.h"
#include<iostream>
using namespace cv;
using namespace std;
//引导滤波器
Mat guidedFilter(Mat& srcMat, Mat& guidedMat, int radius, double eps)
{
    srcMat.convertTo(srcMat, CV_64FC1);
    guidedMat.convertTo(guidedMat, CV_64FC1);
    // 计算均值
    Mat mean_p, mean_I, mean_Ip, mean_II;
    boxFilter(srcMat, mean_p, CV_64FC1, Size(radius, radius));                      // 生成待滤波图像均值mean_p 
    boxFilter(guidedMat, mean_I, CV_64FC1, Size(radius, radius));                   // 生成引导图像均值mean_I   
    boxFilter(srcMat.mul(guidedMat), mean_Ip, CV_64FC1, Size(radius, radius));      // 生成互相关均值mean_Ip
    boxFilter(guidedMat.mul(guidedMat), mean_II, CV_64FC1, Size(radius, radius));   // 生成引导图像自相关均值mean_II
    // 计算相关系数、Ip的协方差cov和I的方差var------------------
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_II - mean_I.mul(mean_I);
    // 计算参数系数a、b
    Mat a = cov_Ip / (var_I + eps);
    Mat b = mean_p - a.mul(mean_I);
    // 计算系数a、b的均值
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_64FC1, Size(radius, radius));
    boxFilter(b, mean_b, CV_64FC1, Size(radius, radius));
    // 生成输出矩阵
    Mat dstImage = mean_a.mul(srcMat) + mean_b;
    return dstImage;
}

int main() {

	int ability;
    int a, b;
    printf("1.图像灰度、彩度生成 2.添加高斯噪声");
	scanf_s("%d", &ability);
	Mat image = imread("C:/Users/oscar/Pictures/ODYSSEY1.png");
    Mat dst, dst2;
	if (image.empty()) {
		printf("打不开");
		return -1;
	}
	if (ability == 1) {
       namedWindow("展示", WINDOW_FREERATIO);
       imshow("test", image);
       Quickdemo qd;
       qd.colorSpace_Demo(image);
       cv::waitKey(0);
	}
    else if (ability == 2) {
        Quickdemo qd;
        qd.noise_img(image);
        imshow("高斯噪声", image); 
        cv::waitKey(0);
        imwrite("C:/Users/oscar/Pictures/Opencv/Gaussin.png",image );
    }
    else if (ability == 3) {
        GaussianBlur(image, dst, Size(5, 5), 0);
        namedWindow("高斯滤波", WINDOW_FREERATIO);
        imshow("高斯滤波", dst);
        cv::waitKey(0);
        imwrite("C:/Users/oscar/Pictures/Opencv/BlurGaussin.png", dst);
    }
    else if (ability == 4) {
        int x; // 起始点的x坐标
        int y;  // 起始点的y坐标
        int width;  // 裁剪区域的宽度
        int height; // 裁剪区域的高度
        scanf_s("%d %d %d %d", &x, &y, &width, &height);
        // 裁剪图像
        Rect region_of_interest(x, y, width, height);
        Mat cropped_image = image(region_of_interest);
        // 显示裁剪后的图像
        imshow("Cropped Image", cropped_image);
        cv::waitKey(0);
        imwrite("C:/Users/oscar/Pictures/Opencv/cropped_image.png", cropped_image);
    }
    else if (ability == 5) {
        int c;
        printf("1.旋转 2.缩放 3.对称");
        scanf_s("%d", &c);
        if (c == 1) {
            cv::Point2f center(image.cols / 2.0f, image.rows / 2.0f);

            // 旋转角度，以度为单位
            double angle = 45.0;

            // 获取旋转矩阵
            cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);

            // 旋转图像
            cv::Mat dst;
            cv::warpAffine(image, dst, rotMat, image.size());

            // 显示旋转后的图像
            cv::imshow("Rotated Image", dst);
            imwrite("C:/Users/oscar/Pictures/Opencv/Rotated Image.png", dst);
            cv::waitKey(0);
        }
        else if (c == 2) {
            // 创建一个Mat对象用于存储缩放后的图像
            cv::Mat resizedImage;

            // 设定缩放后图像的大小
            cv::Size size(400, 480); // 假设我们想要的新尺寸是640x480

            // 缩放图像
            cv::resize(image, resizedImage, size);

            // 保存缩放后的图像
            cv::imshow("resized Image", resizedImage);
            imwrite("C:/Users/oscar/Pictures/Opencv/resizedImage.png", resizedImage);
            cv::waitKey(0);
        }
        else if (c == 3) {
            cv::Size size = image.size();

            // 设置仿射变换矩阵
            cv::Mat matrix = cv::Mat::eye(2, 3, CV_64F);
            matrix.at<double>(0, 0) = -1;  // 水平翻转
            matrix.at<double>(0, 2) = size.width;  // 水平翻转后的平移，使之关于x轴对称

            // 执行仿射变换
            cv::Mat dst;
            cv::warpAffine(image, dst, matrix, size);

            // 显示结果
            cv::imshow("Original Image", image);
            cv::imshow("Symmetric Image", dst);
            imwrite("C:/Users/oscar/Pictures/Opencv/Symmetric Image.png", dst);
            cv::waitKey(0);
        }
    }
    else if (ability == 6) {
        int d;
        printf("1.应用直方图均衡化 2.调整亮度和对比度 3.调整饱和度");
        scanf_s("%d", &d);
        if (d == 1) {
            Mat stc_bgr[3];

            // 拆通道
            split(image, stc_bgr);
            for (int i = 0; i < 3; i++)
            {
                equalizeHist(stc_bgr[i], stc_bgr[i]);
            }
            //   合并通道
            merge(stc_bgr, 3, dst);
            imshow("增强后的图像", dst);
            imwrite("C:/Users/oscar/Pictures/Opencv/equalizeHistImage.png", dst);
            cv::waitKey(0);

        }
        else if (d == 2) {


            double alpha = 1.5; // 对比度调整因子
            int beta = 50; // 亮度调整值

            // 调整对比度和亮度
            cv::Mat contrastBrightImg;
            cv::convertScaleAbs(image, contrastBrightImg, alpha, beta);
            cv::imshow("调整后的图像", contrastBrightImg);
            imwrite("C:/Users/oscar/Pictures/Opencv/contrastBrightImg.png", contrastBrightImg);
            cv::waitKey(0);

        }
        else if (d == 3) {
            Mat hsvImage;
            cvtColor(image, hsvImage, COLOR_BGR2HSV);

            // 调整饱和度
            float saturationScale = 5.0; // 调整饱和度的比例，可根据需要调整
            vector<Mat> channels;
            split(hsvImage, channels);
            channels[1] = channels[1] * saturationScale;
            merge(channels, hsvImage);

            // 将图像转换回BGR色彩空间
            Mat adjustedImage;
            cvtColor(hsvImage, adjustedImage, COLOR_HSV2BGR);

            // 显示原始图像和调整后的图像
            imshow("调整后的图像", adjustedImage);
            imwrite("C:/Users/oscar/Pictures/Opencv/adjustedImage.png", adjustedImage);
            cv::waitKey(0);
        }
    }
    else if (ability == 7) {
        int e;
        printf("1.添加边框 2.虚化图片 3.拼接图片 4.马赛克 5.倒影图");
        scanf_s("%d", &e);
        if (e == 1) {
            cv::Mat borderedImg(image.rows + 20, image.cols + 20, image.type(), cv::Scalar(2, 255, 255));
 
            // 将原图放置在新图片的中心
             cv::Rect roi(10, 10, image.cols, image.rows);
          image.copyTo(borderedImg(roi));
         imshow("添加后的图像", borderedImg);
         imwrite("C:/Users/oscar/Pictures/Opencv/borderedImg.png", borderedImg);
          cv::waitKey(0);


        }
        else if (e == 2) {
            Mat resultMat;
            Mat vSrcImage[3], vResultImage[3];

            split(image, vSrcImage);
            for (int i = 0; i < 3; i++)
            {
                Mat tempImage;
                vSrcImage[i].convertTo(tempImage, CV_64FC1, 1.0 / 255.0);
                Mat cloneImage = tempImage.clone();
                Mat resultImage = guidedFilter(tempImage, cloneImage, 5, 0.3);
                vResultImage[i] = resultImage;
            }
            // 将分通道导向滤波后结果合并
            merge(vResultImage, 3, resultMat);
            imshow("背景虚化特效", resultMat);
            imwrite("C:/Users/oscar/Pictures/Opencv/resultMatImg.png", resultMat);
            cv::waitKey(0);
        }
        else if (e == 3) {
            cv::Mat image2 = cv::imread("C:/Users/oscar/Pictures/ODYSSEY1.png");
            if (image.empty() || image2.empty()) {
                std::cout << "图片读取失败" << std::endl;
                return -1;
            }
            if (image.rows != image2.rows || image.cols != image2.cols) {
                std::cout << "图片尺寸不匹配" << std::endl;
                return -1;
            }

            // 创建一个新的Mat对象用于存储拼接后的图片
            cv::Mat result = cv::Mat::zeros(image.size(), image.type());

            // 拼接图片
            // 横向拼接
            cv::hconcat(image, image2, result);

            // 或者纵向拼接
            // cv::vconcat(image1, image2, result);

            // 显示结果
            cv::imshow("拼接结果", result);
            imwrite("C:/Users/oscar/Pictures/Opencv/resultImg.png", result);
            cv::waitKey(0);
        }
        else if (e == 4) {
            int blockSize = 5; // 可根据需要调整块的大小

            // 遍历图像并应用马赛克变换
            for (int y = 0; y < image.rows; y += blockSize) {
                for (int x = 0; x < image.cols; x += blockSize) {
                    // 计算当前块的范围
                    int maxX = min(x + blockSize, image.cols);
                    int maxY = min(y + blockSize, image.rows);

                    // 计算当前块的平均颜色
                    Rect roi(x, y, maxX - x, maxY - y);
                    Scalar avgColor = mean(image(roi));

                    // 将当前块的像素值设置为平均颜色
                    image(roi).setTo(avgColor);
                }
            }

            // 显示马赛克变换后的图像
            imshow("Original Image", image);
            imwrite("C:/Users/oscar/Pictures/Opencv/Original Image.png", image);
            cv::waitKey(0);

        }
        else if (e == 5) {


            Mat flippedImage;
            flip(image, flippedImage, 1);

            // 合并原始图像和倒影图像
            Mat reflectedImage;
            vconcat(image, flippedImage, reflectedImage);

            // 显示结果
            namedWindow("Reflected Image", WINDOW_AUTOSIZE);
            imshow("Reflected Image", reflectedImage);
            imwrite("C:/Users/oscar/Pictures/Opencv/Reflected Image.png", reflectedImage);
            cv::waitKey(0);
        }
    }
    
    return 0;
}
