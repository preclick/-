#include"quickopencv.h"
void Quickdemo::colorSpace_Demo(Mat &image) {
	//灰度，彩度图像生成
	Mat gray, hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("HSV", hsv);
	imshow("GRAY", gray);
	imwrite("C:/Users/oscar/Pictures/Opencv/hsv.png", hsv);
	imwrite("C:/Users/oscar/Pictures/Opencv/gray.png", gray);

}
void Quickdemo::noise_img(Mat& image)
{
	//产生高斯噪声
	Mat noise = Mat::zeros(image.size(), image.type());
	randn(noise, (25, 25, 25), (30, 30, 30));//生成噪声图像
	Mat dst;
	add(noise, image, dst);
}
void Quickdemo::adjustSaturation(cv::Mat& image, double saturationFactor) {
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

	// Split the image into its channels
	std::vector<cv::Mat> hsvChannels;
	cv::split(hsvImage, hsvChannels);

	// Adjust the saturation channel
	hsvChannels[1] *= saturationFactor;

	// Merge the channels back into a single image
	cv::merge(hsvChannels, hsvImage);

	// Convert the image back to BGR color space
	cv::cvtColor(hsvImage, image, cv::COLOR_HSV2BGR);
}


//cv::Mat borderedImg(image.rows + 20, image.cols + 20, image.type(), cv::Scalar(2, 255, 255));
//
//// 将原图放置在新图片的中心
//cv::Rect roi(10, 10, image.cols, image.rows);
//image.copyTo(borderedImg(roi));
//imshow("添加后的图像", borderedImg);
//imwrite("C:/Users/oscar/Pictures/Opencv/borderedImg.png", borderedImg);
//cv::waitKey(0);