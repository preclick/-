#pragma once
#include<opencv2/opencv.hpp>
using namespace cv;
class Quickdemo {
 public :
	void colorSpace_Demo(Mat &image);
	void noise_img(Mat& image);
	void adjustSaturation(cv::Mat& image, double saturationFactor);
};

