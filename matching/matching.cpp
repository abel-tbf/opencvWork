#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/video/tracking.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/legacy/legacy.hpp>
#include<iostream>
#include<vector>

using namespace std;

const double PI = std::atan(1.0)*4;

class HarrisDetector {
private:
	// 32-bit float image of corner strength
	cv::Mat cornerStrength;
	// 32-bit float image of thresholded corners
	cv::Mat cornerTh;
	//imge of local maxima (internal)
	cv::Mat localMax;
	// size of neighborhood for derivatives smoothing
	int neighbourhood;
	// aperture for gradient computation
	int aperture;
	// Harris parameter
	double k;
	// maximum stength for thershold computation
	double maxStrength;
	// calculated threshold (internal)
	double threshold;
	// size of neighborhood for non-max suppresion
	int nonMaxSize;
	// kernel for non-max suppression
	cv::Mat kernel;
public:
	HarrisDetector() : neighbourhood(3), aperture(3),
						k(0.01), maxStrength(0.0),
						threshold(0.01), nonMaxSize(3) {
		// create kernel used in non-maxima suppresion
		setLocalMaxWindowSize(nonMaxSize);
	}
	
	// Create kernel used in non-maxima suppression
	void setLocalMaxWindowSize(int size){
		
		nonMaxSize = size;
		kernel.create(nonMaxSize, nonMaxSize, CV_8U);
	}
	
	// Compute Harris corners
	void detect(const cv::Mat& image){
		// Harris computation
		cv::cornerHarris(image, cornerStrength,
				neighbourhood, // neighborhood size
				aperture, // aperture size
				k); // Harris parameter
		// internal threshold computation
		double minStrength; // not used
		cv::minMaxLoc(cornerStrength,
			&minStrength, &maxStrength);
		// local maxima detection
		cv::Mat dilated; // temporary image
		cv::dilate(cornerStrength,dilated,cv::Mat());
		cv::compare(cornerStrength, dilated,
					localMax, cv::CMP_EQ);
	}
	
	// Get the corner map from the computed Harris values
	cv::Mat getCornerMap(double qualityLevel){
		cv::Mat cornerMap;
		// thresholding the corner strength
		threshold = qualityLevel * maxStrength;
		cv::threshold(cornerStrength, cornerTh,
					threshold, 255, cv::THRESH_BINARY);
		// convert to 8-bit image
		cornerTh.convertTo(cornerMap, CV_8U);
		// non-maxima suppresion
		cv::bitwise_and(cornerMap, localMax, cornerMap);
		return cornerMap;
	}
	
	// Get the feature points from the computed Harris values
	void getCorners(std::vector<cv::Point> &points,
					double qualityLevel){
		// Get the corner map
		cv::Mat cornerMap = getCornerMap(qualityLevel);
		// Get the corners
		getCorners(points, cornerMap);	
	}
	
	// Get the feature points from the computed corner map
	void getCorners(std::vector<cv::Point> &points,
					const cv::Mat& cornerMap){
		// Iterate over the pixels to obtain all features
		for(int y = 0; y < cornerMap.rows; y++){
			const uchar* rowPtr = cornerMap.ptr<uchar>(y);
			for(int x = 0; x < cornerMap.cols; x++){
				// if it is a feature point
				if(rowPtr[x]){
					points.push_back(cv::Point(x,y));
				}
			}
		}	
	}
	
	// Draw circles at feature point locations on an image
	void drawOnImage(cv::Mat &image,
		const std::vector<cv::Point> &points,
		cv::Scalar color = cv::Scalar(255,255,255),
		int radius = 3, int thickness = 2) {
			
		std::vector<cv::Point>::const_iterator it = 
										points.begin();
		// for all corners
		while(it != points.end()){
			// draw a circle at each corner location
			cv::circle(image, *it, radius, color, thickness);
			++it;
		}
	}
};

void harrisCornerTest(){
	char imgPath[] = "../images/tower.jpg";
	cv::Mat image = cv::imread(imgPath,0);
	
	// Detect Harris Corners
	cv::Mat cornerStrength;
	cv::cornerHarris(image, cornerStrength,
					3, // neightborhood size
					3, // aperture size
					0.01); // Harris parameter
	// threshold the corner strngths
	cv::Mat harrisCorners;
	double threshold = 0.0001;
	cv::threshold(cornerStrength, harrisCorners,
				threshold,255,cv::THRESH_BINARY_INV);
				
	cv::namedWindow("Harris Corners Map");
	cv::imshow("Harris Corners Map",harrisCorners);
}

void harrisDetectorTest(){
	char imgPath[] = "../images/tower.jpg";
	cv::Mat image = cv::imread(imgPath,0);
	
	// Create Harris detector instance
	HarrisDetector harris;
	// Compute Harris values
	harris.detect(image);
	// Detect Harris corners
	std::vector<cv::Point> pts;
	harris.getCorners(pts,0.01);
	// Draw Harris corners
	harris.drawOnImage(image,pts);
	
	cv::namedWindow("Harris Corners");
	cv::imshow("Harris Corners",image);	
}

void goodFeaturesTest(){
	char imgPath[] = "../images/tower.jpg";
	cv::Mat image = cv::imread(imgPath,0);
	
	// Compute good features to track
	std::vector<cv::Point2f> corners;
	cv::goodFeaturesToTrack(image, corners,
		500, // maximum number of corners to be returned
		0.01, // quality level
		10); // minimum allowed distance between points
	
	
	
	// for all corners
	std::vector<cv::Point2f>::const_iterator it = corners.begin();
	while(it != corners.end()){
		// draw a circle at each corner location
		cv::circle(image,*it,3,cv::Scalar(255,255,255),2);
		++it;
	}
	
	// Display the corner
	cv::namedWindow("Good Features to Track");
	cv::imshow("Good Features to Track",image);
}

void goodFeaturesDetectorTest(){
	char imgPath[] = "../images/tower.jpg";
	cv::Mat image = cv::imread(imgPath,0);	
	
	// vector of keypoints
	std::vector<cv::KeyPoint> keypoints;
	// Construction of the Good Feature to Track detector
	cv::GoodFeaturesToTrackDetector gftt(
		500, // maximum number of corners to be returned
		0.01, // quality level
		10.0); // minimum allowed distance between points
	// point detection using FeatureDetector method
	gftt.detect(image,keypoints);
	
	cv::drawKeypoints(image, // original image
		keypoints, // vector of keypoints
		image, // the resulting image
		cv::Scalar(255,255,255), // color of the points
		cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //drawing flag
	
	// Display the corners
	cv::namedWindow("Good Features to Track Detector");
	cv::imshow("Good Features to Track Detector",image);
}

void fastFeatureTest(){
	char imgPath[] = "../images/tower.jpg";
	cv::Mat image = cv::imread(imgPath,0);
	
	// vecotr of keypoints
	std::vector<cv::KeyPoint> keypoints;
	// Construction of the Fast feature detector object
	cv::FastFeatureDetector fast(
		40); // threshold for detection
	// feature point detection
	fast.detect(image,keypoints);
	
	cv::drawKeypoints(image, // original image
		keypoints, // vector of keypoints
		image, // the outpur image
		cv::Scalar(255,255,255), // keypoint color
		cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);  // drawing flag
		
	cv::namedWindow("FAST Features");
	cv::imshow("FAST Features", image);
}

// SURF  AND SIFT ARE NOT FREE
//~ void surfFeatureTest(){
	//~ char imgPath[] = "../images/tower.jpg";
	//~ cv::Mat image = cv::imread(imgPath,0);
	//~ 
	//~ // vector of keypoints
	//~ std::vector<cv::KeyPoint> keypoints;
	//~ // Construct the SURF feature detector object
	//~ cv::SurfFeatureDetector surf(
		//~ 2500.0); // threshold
	//~ // Detect the SURF feature
	//~ surf.detect(image, keypoints);
	//~ 
	//~ // Draw the keypoint with scale and orientation information
	//~ cv::Mat featureImage;
	//~ cv::drawKeypoints(image, // original image
		//~ keypoints, // vecter of keypoints
		//~ featureImage, // the resulting image
		//~ cv::Scalar(255,255,255), // color of the points
		//~ cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); // flags
	//~ 
		//~ 
	//~ cv::namedWindow("SURF Features");
	//~ cv::imshow("SURF Features", image);	
//~ }

// We use orb instead of surf or sift
void orbFeatureTest(){
	char imgPath[] = "../images/tower.jpg";
	cv::Mat image = cv::imread(imgPath,0);
	
	// vector of keypoints
	std::vector<cv::KeyPoint> keypoints;
	// Construct the SURF feature detector object
	cv::OrbFeatureDetector surf(
		250.0); // threshold
	// Detect the SURF feature
	surf.detect(image, keypoints);
	
	// Draw the keypoint with scale and orientation information
	cv::Mat featureImage;
	cv::drawKeypoints(image, // original image
		keypoints, // vecter of keypoints
		featureImage, // the resulting image
		cv::Scalar(255,255,255), // color of the points
		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); // flags
	
		
	cv::namedWindow("SURF Features");
	cv::imshow("SURF Features", featureImage);	
}

void matchesTest(){
	char imgPath1[] = "../images/abelFace1.jpg";
	cv::Mat image1 = cv::imread(imgPath1,0);
	char imgPath2[] = "../images/abelFace2.jpg";	
	cv::Mat image2 = cv::imread(imgPath2,0);
	
	// vector of keypoints
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	// Construction of the SURF descriptor extractor
	cv::OrbFeatureDetector surf(3000);

	// Detection of the SURF Features
	surf.detect(image1,keypoints1);
	surf.detect(image2, keypoints2);
	
	// Detection of the SURF features
	cv::Mat descriptors1;
	cv::Mat descriptors2;
	surf.compute(image1,keypoints1, descriptors1);
	surf.compute(image2,keypoints2, descriptors2);
	
	// Construction of the matcher
	cv::BruteForceMatcher<cv::L2<float> > matcher;
	
	// Match the two image descriptors
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);
	
	std::nth_element(matches.begin(), // initial position
		matches.begin() + 15, // position of the sorted element
		matches.end()); // end position
	// remove all elements after the 25th
	matches.erase(matches.begin() + 16, matches.end());
	
	cv::Mat imageMatches;
	cv::drawMatches(
		image1, keypoints1, // 1st image and its keypoints
		image2, keypoints2, // 2nd image and its keypoints
		matches, // the matches
		imageMatches, // the image produced
		cv::Scalar(255, 255, 255)); // color of the lines
		//~ 
	cv::namedWindow("Matches");
	cv::imshow("Matches", imageMatches);
}

int main(){

	//~ harrisCornerTest();
	//~ harrisDetectorTest();
	//~ goodFeaturesTest();
	//~ goodFeaturesDetectorTest();
	//~ fastFeatureTest();
	//~ orbFeatureTest();
	matchesTest();
	cv::waitKey();
	return 0;
}
