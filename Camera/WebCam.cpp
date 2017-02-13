/* 2016/10/26 14:24
* WebCam version 02
* 讀取 WebCam
* 灰階 WebCam
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

int main()
{
	Mat image, gray_image;

	VideoCapture cap;      //capture的宣告
	cap.open(0);           //0 為預設camera

	while (cap.isOpened())  //確認camera能開啓
	{
		cap >> image;        //截取影像到image裡方法1
		//cap.read(image); //截取影像到image裡方法2
		//以上兩種方法都可以用，實測過沒問題!
		cvtColor(image, gray_image, CV_RGB2GRAY);  //截取影像轉成灰階影像
		imshow("Webcam live", image);	//顯示原始即時影像
		imshow("Webcam gray", gray_image);	//顯示灰階即時影像

		waitKey(33);//避免CPU負荷，給點delay時間
		//實際上一般webcam的framerate差不多33ms
	}

	return 0;
}