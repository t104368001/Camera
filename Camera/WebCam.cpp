/* 2016/11/09 22:18
* WebCam version 04
* 讀取 WebCam
* 灰階 WebCam
* 皮膚二值化
* 鍵盤輸入
* mThread
*/

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <thread>

#define IMAGE_MAX 255
#define IMAGE_MIN 0 
#define IMAGE_ROWS 640
#define IMAGE_COLS 480
#define FILE_NAME "Img"
#define FILE_EXTENSION ".bmp"

using namespace cv;
using namespace std;

String int2str(int);
Mat skin_binarization(Mat);
void getKey(int, int*, bool*, Mat);

int main()
{
	Mat image, gray_image, ycbcr_image, skin_binarization_image(IMAGE_COLS, IMAGE_ROWS, CV_8U, Scalar(0)), binarization_image(IMAGE_COLS, IMAGE_ROWS, CV_8U, Scalar(0));

	VideoCapture cap;      //capture的宣告
	cap.open(0);           //0 為預設camera
	int count = 0;
	bool flag = true;

	while (cap.isOpened() && flag)  //確認camera能開啓
	{
		cap >> image;        //截取影像到image裡方法1
		//cap.read(image); //截取影像到image裡方法2
		//以上兩種方法都可以用，實測過沒問題!

		//cvtColor(image, gray_image, CV_RGB2GRAY);  //截取影像轉成灰階影像
		cvtColor(image, ycbcr_image, COLOR_BGR2YCrCb);	//截取影像轉成YCbCr影像

		skin_binarization_image = skin_binarization(ycbcr_image);

		thread mThread(getKey, cvWaitKey(20), &count, &flag, skin_binarization_image);
		mThread.join();

		imshow("Webcam live", image);	//顯示原始即時影像
		//imshow("Webcam gray", gray_image);	//顯示灰階即時影像
		//imshow("Webcam gray", ycbcr_image);		//顯示YCbCr即時影像
		imshow("Webcam skin binarization", skin_binarization_image);		//顯示二值化即時影像

		waitKey(33);//避免CPU負荷，給點delay時間
		//實際上一般webcam的framerate差不多33ms
	}

	return 0;
}

/* 數字轉字串 */
String int2str(int number){
	stringstream Tostring;
	Tostring << number;
	return Tostring.str();
}

/* 求皮膚二值化 */
Mat skin_binarization(Mat imageSource){
	Mat imageresult(IMAGE_COLS, IMAGE_ROWS, CV_8U, Scalar(0));
	for (int col(0); col < imageSource.cols; ++col)
	{
		for (int row(0); row < imageSource.rows; ++row)
		{
			cv::Vec3f pixel = imageSource.at<cv::Vec3b>(row, col);
			if ((60.0 <= pixel[0] && pixel[0] <= 255.0) && (100.0 <= pixel[2] && pixel[2] <= 125.0) && (135.0 <= pixel[1] && pixel[1] <= 170.0)){
				imageresult.at<uchar>(row, col) = IMAGE_MAX;
			}
			else {
				imageresult.at<uchar>(row, col) = IMAGE_MIN;
			}
		}
	}
	return imageresult;
}

void getKey(int key, int* count, bool *flag, Mat image){
	String fileName = "";
	/* get user press button wait 20 milliseconds */
	switch ((char)key){
		/* user not press button */
	case -1:
		break;
		/* exit if user press 'Esc' */
	case 27:
		*flag = false;
		break;
		/* get frame if user press 'space' */
	case 32:
		*count = *count + 1;
		fileName = FILE_NAME + int2str(*count) + FILE_EXTENSION;
		imwrite(fileName, image);
		break;
		/* reset count if user press 'z' */
	case 122:
		*count = 0;
		break;
		/* user press other button */
	default:
		printf("%d\n", (int)key);
	}
}