/* 2016/11/09 20:47
* WebCam version 03
* 讀取 WebCam
* 灰階 WebCam
* 皮膚二值化
*/

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define MAX 255
#define MIN 0 
#define IMAGE_ROWS 640
#define IMAGE_COLS 480

using namespace cv;
using namespace std;

int main()
{
	Mat image, gray_image, ycbcr_image, skin_binarization_image(IMAGE_COLS, IMAGE_ROWS, CV_8U, Scalar(0)), binarization_image(IMAGE_COLS, IMAGE_ROWS, CV_8U, Scalar(0));

	VideoCapture cap;      //capture的宣告
	cap.open(0);           //0 為預設camera

	while (cap.isOpened())  //確認camera能開啓
	{
		cap >> image;        //截取影像到image裡方法1
		//cap.read(image); //截取影像到image裡方法2
		//以上兩種方法都可以用，實測過沒問題!

		//cvtColor(image, gray_image, CV_RGB2GRAY);  //截取影像轉成灰階影像
		cvtColor(image, ycbcr_image, COLOR_BGR2YCrCb);	//截取影像轉成YCbCr影像

		/* 求皮膚二值化 */
		for (int col(0); col < ycbcr_image.cols; ++col)
		{
			for (int row(0); row < ycbcr_image.rows; ++row)
			{
				cv::Vec3f pixel = ycbcr_image.at<cv::Vec3b>(row, col);
				//std::cout << "row: " << row << "; col: " << col << "; Y: " << pixel[0] << "; Cr: " << pixel[1] << "; Cb: " << pixel[2] << std::endl;
				if ((60.0 <= pixel[0] && pixel[0] <= 255.0) && (100.0 <= pixel[2] && pixel[2] <= 125.0) && (135.0 <= pixel[1] && pixel[1] <= 170.0)){
					skin_binarization_image.at<uchar>(row, col) = MAX;
				}
				else {
					skin_binarization_image.at<uchar>(row, col) = MIN;
				}
			}
		}

		imshow("Webcam live", image);	//顯示原始即時影像
		//imshow("Webcam gray", gray_image);	//顯示灰階即時影像
		//imshow("Webcam gray", ycbcr_image);		//顯示YCbCr即時影像
		imshow("Webcam gray", skin_binarization_image);		//顯示二值化即時影像

		waitKey(33);//避免CPU負荷，給點delay時間
		//實際上一般webcam的framerate差不多33ms
	}

	return 0;
}