#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include <iomanip>

using namespace cv;
using namespace std;

int main(int argc, const char** argv)
{
	Mat img = imread("Leaf2.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data  and store it in 'img'


	//===============================================================================================
	//namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
	//namedWindow("Patch", CV_WINDOW_AUTOSIZE);

	int iLowH = 31;
	int iHighH = 80;

	int iLowS = 10;
	int iHighS = 255;

	int iLowV = 44;
	int iHighV = 255;

	//============= Detecting the color patch ============


	int pLowH = 0;
	int pHighH = 22;

	int pLowS = 66;
	int pHighS = 208;

	int pLowV = 38;
	int pHighV = 250;


	//Create trackbars in "Control" window
	createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	createTrackbar("HighH", "Control", &iHighH, 179);

	createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	createTrackbar("HighS", "Control", &iHighS, 255);

	createTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
	createTrackbar("HighV", "Control", &iHighV, 255);


	//=============== Trackbars for the Patch in "Patch" window ================
	createTrackbar("LHue", "Patch", &pLowH, 179); //Hue (0 - 179)
	createTrackbar("HHue", "Patch", &pHighH, 179);

	createTrackbar("LSat", "Patch", &pLowS, 255); //Saturation (0 - 255)
	createTrackbar("HSat", "Patch", &pHighS, 255);

	createTrackbar("LVal", "Patch", &pLowV, 255);//Value (0 - 255)
	createTrackbar("HVal", "Patch", &pHighV, 255);



	while (true)
	{

		//===================== Applying filters for the leaf ==========================

		Mat imgHSV;

		cvtColor(img, imgHSV, COLOR_BGR2HSV); //Convert the image from BGR to HSV

		Mat imgThresholded;

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

		//morphological opening (removes small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//morphological closing (removes small holes from the foreground)
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));



		//===================== Applying filters for the Patch ==========================

		Mat imgHSV2;

		cvtColor(img, imgHSV2, COLOR_BGR2HSV);

		Mat imgPatch;

		inRange(imgHSV2, Scalar(pLowH, pLowS, pLowV), Scalar(pHighH, pHighS, pHighV), imgPatch);

		//morphological opening (removes small objects from the foreground)
		erode(imgPatch, imgPatch, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgPatch, imgPatch, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//morphological closing (removes small holes from the foreground)
		dilate(imgPatch, imgPatch, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgPatch, imgPatch, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));


		//============== create a mask image to count all leaf pixels  ==============


		// Transform source image to gray if it is not
		Mat gray;

		if (img.channels() == 3)
		{
			cvtColor(img, gray, CV_BGR2GRAY);
		}
		else
		{
			gray = img;
		}

		// Show gray image
		//imshow("gray", gray);

		// Transform it to binary and invert it. White on black is needed.
		Mat bw;
		threshold(gray, bw, 40, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

		vector<Vec4i> hierarchy;
		vector<vector<Point> > contours;

		// extract only the external blob
		findContours(bw.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		Mat mask = Mat::zeros(bw.size(), CV_8UC1);

		// draw the contours as a solid blob, and create a mask of the cloud
		for (size_t i = 0; i < contours.size(); i++)
			drawContours(mask, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());



		vector<Point> leaf_pixels;
		cv::findNonZero(mask, leaf_pixels);


		//=============== End of segmentation ==================



		///==============  Pixel Calculations  =============================

		// ======================== Count diseased pixels

		vector<Point> white_pixels;
		cv::findNonZero(imgPatch, white_pixels);

		cout << "Count all diseased pixels: " << white_pixels.size() << endl; // amount of diseased  pixels is returned from the size




		//================ Count leaf pixels  ==============

		cout << "Count all leaf pixels: " << leaf_pixels.size() << endl; // amount of all leaf pixels is returned from the size

		float all = leaf_pixels.size();
		float diseases = white_pixels.size();

		float per = ((float)diseases / (float)all) * 100;

		cout << fixed << setprecision(3);
		cout << "percentage  of diesase : " << per << "%" << endl;

		if (diseases > 0)
		{
			cout << "This is a Diseased leaf  " << endl;
		}

		else
		{
			cout << "This is a Undiseased leaf " << endl;
		}

		///================ End of Calculations  ========================


		imshow("Original", img); //show the original image
		imwrite("Original.jpg", img); //save the original image

		//imshow("HSV", imgHSV); //show the original image
		imwrite("HSV_leaf.jpg", imgHSV);  //save the HSV image

		imshow("Diseased Image", imgThresholded); //show the diseased binary image
		imwrite("Diseased_Image.jpg", imgThresholded);  //save the diseased binary image

		imshow("Color Patch", imgPatch); //show the thresholded image
		imwrite("Patch.jpg", imgPatch);  //save the Thresholded image

		//imshow("mask", mask);
		imwrite("Mask.jpg", mask);  //save the Thresholded image

		waitKey(0);

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;

			cvDestroyAllWindows();
		}

		//===============================================================================================




	}

	////////////////////////////////////////////////////////////////

	return 0;
}