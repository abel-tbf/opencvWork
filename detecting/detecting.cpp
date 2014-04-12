#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/video/tracking.hpp>
#include<iostream>


using namespace std;

const double PI = std::atan(1.0)*4;

class LineFinder{
private:
	// original image
	cv::Mat img;
	// vector containing the end points
	// of the detected lines
	std::vector<cv::Vec4i> lines;
	// accumulator resolution parameters
	double deltaRho;
	double deltaTheta;
	
	// minimum number of votes that a l ine 
	// must receive before being considered
	int minVote;
	
	// min length for a line
	double minLength;
	
	// max allowed gap along the line
	double maxGap;
	//

public:
	// Default accumulator resolution is 1 pixel by 1 degree
	// no gap, no minimum length
	LineFinder(): deltaRho(1), deltaTheta(PI/180), 
					minVote(10), minLength(0.0), maxGap(0.0){
		
	}
					
	// Set the resolution of the accumulator
	void setAccResolution(double dRho, double dTheta){
		deltaRho = dRho;
		deltaTheta = dTheta;
	}
	// Set the minimum number of votes
	void setMinVote(int minv){
		minVote = minv;
	}
	
	// Set line length and gap
	void setLineLengthAndGap(double length, double gap){
		minLength = length;
		maxGap = gap;
	}
	
	// Apply probabilistic Hough Transform
	std::vector<cv::Vec4i> findLines(cv::Mat& binary){
		lines.clear();
		cv::HoughLinesP(binary, lines,
						deltaRho, deltaTheta, minVote,
						minLength, maxGap);
		return lines;
	}
	
	// Draw the detected lines on an image
	void drawDetectedLines(cv::Mat &image,
			cv::Scalar color = cv::Scalar(255,255,255)){
		// Draw the lines
		std::vector<cv::Vec4i>::const_iterator it2 = lines.begin();
		while(it2!=lines.end()){
			cv::Point pt1((*it2)[0],(*it2)[1]);
			cv::Point pt2((*it2)[2],(*it2)[3]);
			cv::line(image, pt1,pt2,color);
			++it2;
		}
	}
};

class ContentFinder{
private:
	float hranges[2];
	const float* ranges[3];
	int channels[3];
	float threshold;
	cv::MatND histogram;
public:
	ContentFinder() : threshold(-1.0f){
		ranges[0] = hranges; // all channels have same range
		ranges[1] = hranges;
		ranges[2] = hranges;
	}
	
	// Sets the threshold on histogram values [0,1]
	void setThreshold(float t){
		threshold = t;
	}
	
	// Gets the threshold
	float getThreshold(){
		return threshold;
	}
	
	// Sets the reference histogram
	void setHistogram(const cv::MatND& h){
		histogram = h;
		cv::normalize(histogram, histogram, 1.0);
	}
	
	cv::Mat find(const cv::Mat& image){
		cv::Mat result;
		hranges[0] = 0.0;
		hranges[1] = 255.0;
		channels[0] = 0;
		channels[1] = 1;
		channels[2] = 2;
		
		cv::calcBackProject(&image, 1, // input image
			channels, // list of channels used
			histogram, // the histogram we are using
			result, // the resulting backprojection
			ranges, //the range of values
			255.0 // the scaling factor
		);

		// Threshold back projection to obtain a binary image
		if (threshold > 0.0)
			cv::threshold(result, result,
							255 * threshold, 255, cv::THRESH_BINARY);
		return result;
	}
	
	cv::Mat find(const cv::Mat& image,
				float minValue, float maxValue,
				int *channels, int dim){
		cv::Mat result;

		hranges[0] = minValue;
		hranges[1] = maxValue;
		for(int i = 0; i < dim; i++)
			this->channels[i] = channels[i];
					
		cv::calcBackProject(&image, 1, // input image
			channels, // list of channels used
			histogram, // the histogram we are using
			result, // the resulting backprojection
			ranges, //the range of values
			255.0 // the scaling factor
		);

		// Threshold back projection to obtain a binary image
		if (threshold > 0.0)
			cv::threshold(result, result,
							255 * threshold, 255, cv::THRESH_BINARY);
		return result;
	}
};

void lineDetectingTest(){
	
	// Read image
	cv::Mat image= cv::imread("../images/abel.jpg");
	//~ cv::Mat image= cv::imread("../images/build.jpg");
	
	// Create LineFinder instance
	LineFinder finder;
	
	// Set probabilistic Houg parameters
	finder.setLineLengthAndGap(120,20);
	finder.setMinVote(80);
	
	// Apply Canny algorithm
	cv::Mat contours;
	cv::Canny(image, // gray-level image
				contours, // output contours
				125, //low threshold 128, 125
				350); // high threshold 255 , 350
	// Detect lines and draw them
	std::vector<cv::Vec4i> lines = finder.findLines(contours);
	finder.drawDetectedLines(image);
	cv::namedWindow("Detected Lines with HoughP");
	cv::imshow("Detected Lines with HoughP",image);
}

void circleDetectingTest(){
	// open the image
	//~ cv::Mat source = cv::imread("../images/abel.jpg");
	//~ cv::Mat source = cv::imread("../images/abelFace1.jpg");
	//~ cv::Mat source = cv::imread("../images/circle.jpg");
	//~ cv::Mat source = cv::imread("../images/circles.jpg");
	cv::Mat source = cv::imread("../images/carriage.jpg");

	cv::Mat image;
	cv::cvtColor(source,image,CV_BGR2GRAY);
	cv::GaussianBlur(image,image, cv::Size(5,5),1.5);
	
	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(image,circles,CV_HOUGH_GRADIENT,
		2, // accumulator resolution (size of the image / 2)
		50, // minimum distance between two circles
		200, // Canny high threshold
		100, // minimum number of votes
		25,100); // min and max radius
		
	std::vector<cv::Vec3f>::const_iterator itc = circles.begin();
	while(itc!=circles.end()){
		cv::circle(image,
			cv::Point((*itc)[0], (*itc)[1]), // circle centre
			(*itc)[2], // circle radius
			cv::Scalar(255), // color
			2); // thickness
		++itc;
	}
	cv::namedWindow("Detected circles");
	cv::imshow("Detected circles", image);
}

void lineDetectingTest2(){
	// Read image
	//~ char imgPath[]=	"../images/build.jpg";
	//~ char imgPath[]=	"../images/abel.jpg";
	char imgPath[]=	"../images/road.jpg";
	cv::Mat image = cv::imread(imgPath);	
	// Create LineFinder instance
	LineFinder finder;

	//~ // Set probabilistic Houg parameters
	finder.setLineLengthAndGap(120,20);
	finder.setMinVote(80);
	//~ 
	//~ // Apply Canny algorithm
	cv::Mat contours;
	cv::Canny(image, // gray-level image
				contours, // output contours
				125, //low threshold 128, 125
				350); // high threshold 255 , 350
	// Detect lines and draw them
	std::vector<cv::Vec4i> lines = finder.findLines(contours);
	// here we can load the detectected lines
	finder.drawDetectedLines(image);
	cv::namedWindow("Detected Lines");
	cv::imshow("Detected Lines", image);
	
	int n= 0; // we select line 0
	// black image
	cv::Mat oneline(contours.size(), CV_8U, cv::Scalar(0));
	// white line
	cv::line(oneline,
			cv::Point(lines[n][0], lines[n][1]),
			cv::Point(lines[n][2], lines[n][3]),
			cv::Scalar(255),
			2);
	// contours AND white line
	cv::bitwise_and(contours,oneline,oneline);
	
	std::vector<cv::Point> points;
	// Iterate over the pixels to obtain all point positions
	for(int y = 0; y < oneline.rows; y++){
		// row y
		uchar * rowPtr = oneline.ptr<uchar>(y);
		for(int x = 0; x < oneline.cols; x++){
			// column x
			// if on a contour
			if(rowPtr[x]){
				points.push_back(cv::Point(x,y));
			}
		}
	}
	
	cv::Vec4f line;
	cv::fitLine(cv::Mat(points), line,
				CV_DIST_L2, // distance type
				0, // not used with L2 distance
				0.01, 0.01); //accuracy
				
	int x0 = line[2]; // a point on the line
	int y0 = line[3];
	int x1 = x0 - 200 * line[0]; // add a vector of length 200
	int y1 = y0 - 200 * line[1]; // using the unit vector
	cv::Mat drawOneLine = cv::imread(imgPath,0); // the original image is loaded
	cv::line(drawOneLine, cv::Point(x0,y0),cv::Point(x1,y1),
			cv::Scalar(0),3);
			
	cv::namedWindow("One line");
	cv::imshow("One line", drawOneLine);
	
}


void contourTest(){
	char imgPath[] ="../images/group.jpg";
	//~ char imgPath[] ="../images/groupB.jpg";
	//~ char imgPath[] ="../images/abel.jpg";
	// Read input image
	cv::Mat image = cv::imread(imgPath,0); // open in b and w
	cv::Mat thresholded;
	cv::Mat inverted;
	cv::threshold(image,thresholded,90,255,cv::THRESH_BINARY);
	
	 cv::Mat whiteWindow = cv::Mat::ones(thresholded.size(), thresholded.type())*255;
	cv::subtract(whiteWindow,thresholded,inverted);
	
	cv::Mat element5(5,5,CV_8U,cv::Scalar(1));

	cv::Mat closed;
	cv::morphologyEx(inverted,closed,cv::MORPH_CLOSE,element5);
	cv::Mat opened;
	cv::morphologyEx(closed,opened,cv::MORPH_OPEN, element5);
	
	std::vector<std::vector<cv::Point> > contours;
	
	//~ cv::findContours(opened,
		//~ contours, // a vector of contours
		//~ CV_RETR_EXTERNAL, // retrieve the external contours
		//~ CV_CHAIN_APPROX_NONE); // all pixels of each contours
	//~ cv::findContours(opened,
		//~ contours, // a vector of contours
		//~ CV_RETR_LIST, // retrieve the external contours
		//~ CV_CHAIN_APPROX_NONE); // all pixels of each contours
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(opened,
		contours, // a vector of contours
		hierarchy, // hierarchical representation
		CV_RETR_TREE, // retrieve the external contours
		//~ CV_CHAIN_APPROX_NONE); // all pixels of each contours
		CV_RETR_CCOMP); // all pixels of each contours
		
		
	// Draw black contours on a white image
	cv::Mat result(image.size(), CV_8U, cv::Scalar(255));
	cv::drawContours(result,contours,
		-1, // draw all contours
		cv::Scalar(0), // in black
		2); // with a thickness of 2
		
	// Eliminate too shor or too long contours
	unsigned cmin = 100; // minimum contour length
	unsigned cmax = 1000; // maximum contour length
	std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();
	while(itc !=contours.end()){
		if(itc->size() < cmin || itc->size() > cmax)
			itc = contours.erase(itc);
		else
			++itc;
	}
	cv::namedWindow("Contours");
	cv::imshow("Contours",result);
	
	image = cv::imread(imgPath);
	cv::drawContours(image,contours,
		-1, // draw all contours
		cv::Scalar(255), // in black
		2); // with a thickness of 2
	cv::namedWindow("Contours on Animals");
	cv::imshow("Contours on Animals",image);
}

void openCloseTest(){
	char imgPath[] ="../images/group.jpg";
	// Read input image
	cv::Mat image = cv::imread(imgPath,0); // open in b and w
	cv::Mat thresholded;
	cv::Mat inverted;
	cv::threshold(image,thresholded,90,255,cv::THRESH_BINARY);
	
	 cv::Mat whiteWindow = cv::Mat::ones(thresholded.size(), thresholded.type())*255;
	cv::subtract(whiteWindow,thresholded,inverted);
	
	cv::Mat element5(5,5,CV_8U,cv::Scalar(1));

	cv::Mat closed;
	cv::morphologyEx(inverted,closed,cv::MORPH_CLOSE,element5);	
	cv::Mat opened;
	cv::morphologyEx(closed,opened,cv::MORPH_OPEN, element5);

	//~ cv::Mat result;
	//~ // dilate original image
	//~ cv::dilate(inverted,result,cv::Mat());
	//~ // in-place erosion of the dilated image
	//~ cv::erode(result,result,cv::Mat());
	
	cv::namedWindow("Closed and Opened");
	cv::imshow("Closed and Opened",closed);
}

void shapeDescriptionTest(){
	char imgPath[] ="../images/group.jpg";
	//~ char imgPath[] ="../images/groupB.jpg";
	//~ char imgPath[] ="../images/abel.jpg";
	// Read input image
	cv::Mat image = cv::imread(imgPath,0); // open in b and w
	cv::Mat thresholded;
	cv::Mat inverted;
	cv::threshold(image,thresholded,90,255,cv::THRESH_BINARY);
	
	 cv::Mat whiteWindow = cv::Mat::ones(thresholded.size(), thresholded.type())*255;
	cv::subtract(whiteWindow,thresholded,inverted);
	
	cv::Mat element5(5,5,CV_8U,cv::Scalar(1));

	cv::Mat closed;
	cv::morphologyEx(inverted,closed,cv::MORPH_CLOSE,element5);
	cv::Mat opened;
	cv::morphologyEx(closed,opened,cv::MORPH_OPEN, element5);
	
	std::vector<std::vector<cv::Point> > contours;
	
	cv::findContours(opened,
		contours, // a vector of contours
		CV_RETR_EXTERNAL, // retrieve the external contours
		CV_CHAIN_APPROX_NONE); // all pixels of each contours
	//~ cv::findContours(opened,
		//~ contours, // a vector of contours
		//~ CV_RETR_LIST, // retrieve the external contours
		//~ CV_CHAIN_APPROX_NONE); // all pixels of each contours
	//~ std::vector<cv::Vec4i> hierarchy;
	//~ cv::findContours(opened,
		//~ contours, // a vector of contours
		//~ hierarchy, // hierarchical representation
		//~ CV_RETR_TREE, // retrieve the external contours
		//~ CV_CHAIN_APPROX_NONE); // all pixels of each contours
		//~ CV_RETR_CCOMP); // all pixels of each contours

	// Eliminate too shor or too long contours
	unsigned cmin = 50; // minimum contour length
	unsigned cmax = 600; // maximum contour length
	std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();
	while(itc !=contours.end()){
		if(itc->size() < cmin || itc->size() > cmax)
			itc = contours.erase(itc);
		else
			++itc;
	}
	
	// Draw black contours on a white image
	cv::Mat result(image.size(), CV_8U, cv::Scalar(255));
	cv::drawContours(result,contours,
		-1, // draw all contours
		cv::Scalar(0), // in black
		2); // with a thickness of 2

	// testing the bounding box
	cv::Rect r0 = cv::boundingRect(cv::Mat(contours[0]));
	cv::rectangle(result,r0,cv::Scalar(0),2);
	
	// testing enclosing circle
	float radius;
	cv::Point2f center;
	cv::minEnclosingCircle(cv::Mat(contours[1]),center,radius);
	cv::circle(result,cv::Point2f(center),
				static_cast<int>(radius), cv::Scalar(0), 2);

	// testing the approximate polygon ---------------
	std::vector<cv::Point> poly;
	cv::approxPolyDP(cv::Mat(contours[2]), poly,
							5, // accuracy of the approximation
							true); // yes it is a closed shape
	// Iterate over each segment and draw it
	std::vector<cv::Point>::const_iterator itp = poly.begin();
	while (itp !=(poly.end() - 1)) {
		cv::line(result, *itp, *(itp + 1), cv::Scalar(0), 2);
		++itp;
	}
	// last point linked to first point
	cv::line(result,
			*(poly.begin()),
			*(poly.end() - 1), cv::Scalar(20), 2);
	// testing the approximate polygon end ------------------
			
	// testing the convex hull -----------------------
	std::vector<cv::Point> hull;
	cv::convexHull(cv::Mat(contours[3]), hull);
	// Iterate over each segment and draw it
	std::vector<cv::Point>::const_iterator ith = hull.begin();
	while (ith !=(hull.end() - 1)) {
		cv::line(result, *ith, *(ith + 1), cv::Scalar(0), 2);
		++ith;
	}
	// last point linked to first point
	cv::line(result,
			*(hull.begin()),
			*(hull.end() - 1), cv::Scalar(20), 2);	
	// testing the convex hull end -------------------
	
	// testing the moments
	// iterate over all contours
	itc = contours.begin();
	while(itc != contours.end()){
		// compute all moments
		cv::Moments mom = cv::moments(cv::Mat(*itc++));
		// draw mass center
		cv::circle(result,
			// position of mass center converted to integer
			cv::Point(mom.m10/mom.m00,mom.m01/mom.m00),
			2, cv::Scalar(0), 2); // draw black dot
	}
	
	cv::namedWindow("Some Shape descriptors");
	cv::imshow("Some Shape descriptors",result);
	
	image = cv::imread(imgPath);
	cv::drawContours(image,contours,
		-1, // draw all contours
		cv::Scalar(255), // in black
		2); // with a thickness of 2
	cv::namedWindow("Contours on Animals");
	cv::imshow("Contours on Animals",image);
}	

int main(){
	//~ lineDetectingTest();
	//~ circleDetectingTest();
	//~ lineDetectingTest2();
	//~ openCloseTest();
	//~ contourTest();
	shapeDescriptionTest();
	cv::waitKey();
	return 0;
}
