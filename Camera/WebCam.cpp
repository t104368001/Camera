/* 2016/11/11 20:25
* WebCam version 05
* 讀取/灰階 WebCam
* 皮膚二值化
* 鍵盤輸入
* mThread
* using openCV facedetect.cpp
*/

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect.hpp"
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

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip);

string cascadeName = "../../OpenCV 3.0.0/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
string nestedCascadeName = "../../OpenCV 3.0.0/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

int main()
{
	Mat image, gray_image, ycbcr_image, skin_binarization_image(IMAGE_COLS, IMAGE_ROWS, CV_8U, Scalar(0)), binarization_image(IMAGE_COLS, IMAGE_ROWS, CV_8U, Scalar(0));
	CascadeClassifier cascade, nestedCascade;
	VideoCapture cap;      //capture的宣告
	cap.open(0);           //0 為預設camera
	int count = 0;
	bool flag = true;
	bool tryflip = false;
	double scale = 1;

	if (!cascade.load(cascadeName) && !nestedCascade.load(nestedCascadeName))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}

	while (cap.isOpened() && flag)  //確認camera能開?
	{
		cap >> image;        //截取影像到image裡方法1
		//cap.read(image); //截取影像到image裡方法2
		//以上兩種方法都可以用，實測過沒問題!

		//cvtColor(image, gray_image, CV_RGB2GRAY);  //截取影像轉成灰階影像
		//cvtColor(image, ycbcr_image, COLOR_BGR2YCrCb);	//截取影像轉成YCbCr影像

		//skin_binarization_image = skin_binarization(ycbcr_image);

		//thread mThread(getKey, cvWaitKey(20), &count, &flag, skin_binarization_image);
		//mThread.join();
		detectAndDraw(image, cascade, nestedCascade, scale, tryflip);
		//imshow("Webcam live", image);	//顯示原始即時影像
		//imshow("Webcam gray", gray_image);	//顯示灰階即時影像
		//imshow("Webcam gray", ycbcr_image);		//顯示YCbCr即時影像
		//imshow("Webcam skin binarization", skin_binarization_image);		//顯示二值化即時影像

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

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip)
{
	int i = 0;
	double t = 0;
	vector<Rect> faces, faces2;
	const static Scalar colors[] = { CV_RGB(0, 0, 255),
		CV_RGB(0, 128, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(0, 255, 0),
		CV_RGB(255, 128, 0),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 0, 0),
		CV_RGB(255, 0, 255) };
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);

	cvtColor(img, gray, COLOR_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	t = (double)cvGetTickCount();
	cascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		//|CASCADE_FIND_BIGGEST_OBJECT
		//|CASCADE_DO_ROUGH_SEARCH
		| CASCADE_SCALE_IMAGE
		,
		Size(30, 30));
	if (tryflip)
	{
		flip(smallImg, smallImg, 1);
		cascade.detectMultiScale(smallImg, faces2,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			| CASCADE_SCALE_IMAGE
			,
			Size(30, 30));
		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++)
		{
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}
	t = (double)cvGetTickCount() - t;
	printf("detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.));
	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i % 8];
		int radius;

		double aspect_ratio = (double)r->width / r->height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r->x + r->width*0.5)*scale);
			center.y = cvRound((r->y + r->height*0.5)*scale);
			radius = cvRound((r->width + r->height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
		else
			rectangle(img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
			cvPoint(cvRound((r->x + r->width - 1)*scale), cvRound((r->y + r->height - 1)*scale)),
			color, 3, 8, 0);
		if (nestedCascade.empty())
			continue;
		smallImgROI = smallImg(*r);
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			//|CASCADE_DO_CANNY_PRUNING
			| CASCADE_SCALE_IMAGE
			,
			Size(30, 30));
		for (vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++)
		{
			center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
			center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
			radius = cvRound((nr->width + nr->height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
	}
	cv::imshow("result", img);
}