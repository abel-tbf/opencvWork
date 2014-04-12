#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/video/tracking.hpp>
#include<iostream>

using namespace std;

class Histogram1D {
private:
	int histSize[1]; // number of bins
	float hranges[2]; // min and max pixel value
	const float* ranges[1];
	int channels[1]; // only 1 channel used here
public:
	Histogram1D(){
		// Prepare arguments for 1D histogram
		histSize[0] = 256;
		hranges[0] = 0.0;
		hranges[1] = 255.0;
		ranges[0] =hranges;
		channels[0] = 0; // by default, we look at channel 0
	}
	
	// Computes the 1D histogram.
	cv::MatND getHistogram(const cv::Mat &image){
		cv::MatND hist;

		// Compute histogram
		cv::calcHist(&image,
			1, // histogram from 1 image only
			channels, // the channel used
			cv::Mat(), // no mask is used
			hist, // the resulting histogram
			1, // it is a 1D histogram
			histSize, // number of bins
			ranges // pixel value range
		);
		return hist;
	}
				
	// Computes the 1D histogram and returns an image of it.
	cv::Mat getHistogramImage(const cv::Mat &image){
		// Compute histogram first
		cv::MatND hist = getHistogram(image);
		// Get min and max bin values
		double maxVal = 0;
		double minVal = 0;
		cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);
		
		// Image on which to display histogram
		cv::Mat histImg(histSize[0],histSize[0],
						CV_8U, cv::Scalar(255));
		// set highest point at 90% of nbins
		int hpt = static_cast<int>(0.9*histSize[0]);
		// Draw a vertical line for each bin
		for(int h = 0; h < histSize[0]; h++) {
			float binVal = hist.at<float>(h);
			int intensity = static_cast<int>(binVal*hpt/maxVal);
			// This fuction draws a line between 2 points
			cv::line(histImg, cv::Point(h,histSize[0]),
							cv::Point(h,histSize[0] - intensity),
							cv::Scalar::all(0));
		}
		return histImg;
	}
	
	cv::Mat applyLookUp(const cv::Mat& image, // input image
		const cv::Mat& lookup){ // 1x256 uchar matrix
		// the output image
		cv::Mat result;
		// apply lookup table
		cv::LUT(image, lookup, result);
		return result;
	}
	
	cv::Mat stretch(const cv::Mat &image, int minValue = 0){
		// Compute histogram first
		cv::MatND hist = getHistogram(image);
		//find left extremity of the histogram
		int imin = 0;
		for(; imin < histSize[0]; imin++){
			std::cout << hist.at<float>(imin) << std::endl;
			if(hist.at<float>(imin) > minValue)
				break;
		}
		// find right extremity of the histogram
		int imax= histSize[0] - 1;
		for(; imax >= 0; imax--){
			if(hist.at<float>(imax) > minValue)
				break;
		}
		
		// Create lookup table
		int dim(256);
		cv::Mat lookup(
			1, // 1 dimension
			&dim, // 256 entries
			CV_8U); // uchar
		// Build lookup table
		for(int i = 0; i < 256; i++){
			// stretch between imin and imax
			if (i < imin) lookup.at<uchar>(i) = 0;
			else if ( i > imax) lookup.at<uchar>(i)= 255;
			// linear mapping
			else lookup.at<uchar>(i) = static_cast<uchar>(
				255.0*(i - imin)/ (imax-imin) + 0.5);
		}
		
		// Apply lookup table
		cv::Mat result;
		result = applyLookUp(image, lookup);
		return result;
	}
	
	cv::Mat equalize(const cv::Mat &image){
		cv::Mat result;
		cv::equalizeHist(image,result);
		return result;
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

class ColorHistogram {
private:
	int histSize[3];
	float hranges[2];
	const float* ranges[3];
	int channels[3];
public:
	ColorHistogram(){
		// Prepare arguments for a color histogram
		histSize[0] = histSize[1] = histSize[2] = 256;
		hranges[0] = 0.0; // BRG range
		hranges[1] = 255.0; 
		ranges[0] = hranges; // all channels have the same range
		ranges[1] = hranges;
		ranges[2] = hranges;
		channels[0] = 0; // the three channels
		channels[1] = 1;
		channels[2] = 2;
	}
	
	cv::MatND getHistogram(const cv::Mat image){
		cv::MatND hist;
			// Compute histogram
		cv::calcHist(&image,
			1, // histogram of 1 image only
			channels, // the channel used
			cv::Mat(), // no mask is used
			hist, // the resulting histogram
			3, // it is a 3D histogram
			histSize, // number of bins
			ranges // pixel value range
		);
		return hist;
	}
	
	cv::SparseMat getSparseHistogram(const cv::Mat &image){
		cv::SparseMat hist(3,histSize, CV_32F);
		// Compute histogram
		cv::calcHist(&image,
			1, // histogram of 1 image only
			channels, // the channel used
			cv::Mat(), // no mask is used
			hist, // the resulting histogram
			3, // it is a 3D histogram
			histSize, // number of bins
			ranges
		);
		return hist;
	}
	
	cv::Mat colorReduce(const cv::Mat &image, int div=64){
		// clone the image
		cv::Mat result = image.clone();
		int nl = result.rows; // number of lines
		// total numbe of elements per line
		int nc = result.cols * result.channels();
		for(int j = 0; j < nl; j++){
			// get the addres of row j
			uchar * data = result.ptr<uchar>(j);
			for(int i = 0; i < nc; i++){
				// process each pixel ------------------
				data[i] = data[i] / div * div/2;
				// end of pixel processing--------------
			} // end of line
		}
		
		return result;
	}
	
	// Computes the 1D Hue histogram with a mask.
	// BGR source image is converted to HSV
	// Pixels with low saturation are ignored
	cv::MatND getHueHistogram(const cv::Mat &image,
								int minSaturation=0){
		cv::MatND hist;
		// Convert to HSV color space
		cv::Mat hsv;
		cv::cvtColor(image, hsv, CV_BGR2HSV);
		// Mask to be used (or not)
		cv::Mat mask;
		if(minSaturation > 0) {
			// Spliting the 3 channels into 3 images
			std::vector<cv::Mat> v;
			cv::split(hsv, v);
			// Mask out the low satured pixels
			cv::threshold(v[1],mask,minSaturation,255,
								cv::THRESH_BINARY);
		}
		// Prepare arguments for a 1D hue histogram
		hranges[0] = 0.0;
		hranges[1] = 180.0;
		channels[0] = 0; // the hue channel
		// Compute histogram
		cv::calcHist(&hsv, 
			1, // hstogram of 1 image only
			channels, // the channel used
			mask, // binary mask
			hist, // the resulting histogram
			1, // it is a 1D histogram
			histSize, // number of bins
			ranges // pixel value range
		);
		return hist;
	}
};

class ImageComparator{
private:
	cv::Mat reference;
	cv::Mat input;
	cv::MatND refH;
	cv::MatND inputH;
	ColorHistogram hist;
	int div;
public:
	ImageComparator() : div(32) {
		
	}
	// Color reduction factor
	// The comparison will be made on images with
	// color space reduced by this factor in each dimension
	void setColorReduction(int factor) {
		div = factor;
	}
	int getColorReduction(){
		return div;
	}
	void setReferenceImage(const cv::Mat& image){
		reference = hist.colorReduce(image,div);
		refH = hist.getHistogram(reference);
	}
	double compare(const cv::Mat& image) {
		input = hist.colorReduce(image,div);
		inputH = hist.getHistogram(input);
		return cv::compareHist(
						refH,inputH,CV_COMP_INTERSECT);
	}
};

void invertTest(){
	//~ char imgPath[]="../images/abel.jpg";
	//~ char imgPath[]="../images/boldt.jpg";
	char imgPath[]="../images/group.jpg";
	cv::Mat image = cv::imread(imgPath,
									0); // open in b and w
	// Create an image inversion table
	int dim(256);
	cv::Mat lut(1, // 1 dimension
		&dim, // 256 entries
		CV_8U); // uchar
	for (int i = 0; i < 256; i++){
		lut.at<uchar>(i) = 255-i;
	}
	Histogram1D h;
	
	cv::namedWindow("Inverted");
	cv::imshow("Inverted",
				h.applyLookUp(image,lut));
}

void stretchTest(){
	//~ char imgPath[]="../images/abel.jpg";
	//~ char imgPath[]="../images/boldt.jpg";
	char imgPath[]="../images/group.jpg";
	cv::Mat image = cv::imread(imgPath,
									0); // open in b and w

	Histogram1D h;
	// ignore starting and ending bins with less then 100 pixels
	cv::Mat stretched = h.stretch(image,100);
	cv::namedWindow("Stretched");
	cv::imshow("Stretched",
				stretched);
				//~ h.getHistogramImage(stretched));
}

void equalizeTest(){
	//~ char imgPath[]="../images/abel.jpg";
	//~ char imgPath[]="../images/boldt.jpg";
	char imgPath[]="../images/group.jpg";
	cv::Mat image = cv::imread(imgPath,
									0); // open in b and w

	Histogram1D h;

	cv::namedWindow("Equalized");
	cv::imshow("Equalized",
				h.equalize(image));
				//~ h.getHistogramImage((h.equalize(image))));
}

void thresholdTest(){
	char imgPath[]="../images/group.jpg";
	
	// Read input image
	cv::Mat image = cv::imread(imgPath,
								0); // open in b and w
	cv::Mat thresholded;
	cv::threshold(image,thresholded,90,255,cv::THRESH_BINARY);
	cv::namedWindow("Thresholded");
	cv::imshow("Thresholded",thresholded);			
}

void histogramImageTest(){
	//~ char imgPath[]="../images/abel.jpg";
	//~ char imgPath[]="../images/boldt.jpg";
	char imgPath[]="../images/group.jpg";
	
	// Read input image
	cv::Mat image = cv::imread(imgPath,
								0); // open in b and w
	// The histogram object
	Histogram1D h;
	cv::namedWindow("Histogram");
	cv::imshow("Histogram",
				h.getHistogramImage(image));	
}

void histogramTest(){
	//~ char imgPath[]="../images/abel.jpg";
	//~ char imgPath[]="../images/boldt.jpg";
	char imgPath[]="../images/group.jpg";
	
	// Read input image
	cv::Mat image = cv::imread(imgPath,
								0); // open in b and w
	// The histogram object
	Histogram1D h;
	// Compute the histogram
	cv::MatND histo = h.getHistogram(image);
	
	// Loop over each bin
	for(int i = 0; i < 256; i++)
		cout << "Value " << i << " = " <<
				histo.at<float>(i) << endl;
}

void hueTest(){
	// Read reference image
	cv::Mat image= cv::imread("../images/abelFace1.jpg");
	//~ cv::Mat image= cv::imread("../images/baboon1.jpg");
	
	cv::namedWindow("Image 1");
	cv::imshow("Image 1", image);
	// Baboon's face ROI
	cv::Mat imageROI = image(cv::Rect(66,246,206,92));
	// GEt the Hue histogram
	int minSat = 65;
	ColorHistogram hc;
	cv::MatND colorhist = hc.getHueHistogram(imageROI, minSat);
	
	ContentFinder finder;
	finder.setHistogram(colorhist);
	
	image = cv::imread("../images/abel.jpg");
	//~ image = cv::imread("../images/abelFace2.jpg");
	// Display image
	cv::namedWindow("Image 2");
	cv::imshow("Image 2", image);
	// not defined--------
	cv::Mat hsv;
	vector<cv::Mat> v;
	int ch[] = { 0,1,2};
	// not defined end ------
	// Convert to HSV space
	cv::cvtColor(image, hsv, CV_BGR2HSV);
	// Split the imageopencv convert image to 8 bit single channel c++
	cv::split(hsv,v);
	//~ //Identify pixels with low saturation
	cv::threshold(v[1],v[1],minSat,255,cv::THRESH_BINARY);
	
	// Get back-projection of hue histogram
	cv::Mat result = finder.find(hsv,0.0f,180.0f, ch, 1);
	// Eliminate low saturation pixels
	cv::bitwise_and(result, v[1], result);
	
	cv::Rect rect(67,247,206,92);
	cv::rectangle(image,rect, cv::Scalar(0,0,255));
	cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER, 10,0.01);
	cv::meanShift(result,rect,criteria);
	
	cv::namedWindow("Image");
	cv::imshow("Image",
				image);	
}

void finderTest(){
	
	ColorHistogram hc;
	// load color image
	//~ cv::Mat color = cv::imread("../images/abel.jpg");
	cv::Mat color = cv::imread("../images/group.jpg");
	//~ cv::Mat color = cv::imread("../images/boldt.jpg");
	//~ cv::Mat color = cv::imread("../images/sky.jpg");
	// reduce colors
	color = hc.colorReduce(color, 32);
	// blue sky area
	cv::Mat imageROI = color(cv::Rect(0,0,165,75));
	
	cv::MatND hist = hc.getHistogram(imageROI);
	
	ContentFinder finder;
	finder.setHistogram(hist);
	finder.setThreshold(0.05f);
	// Get back-projection of color histogram
	cv::Mat result = finder.find(color);
		cv::namedWindow("Colour Backproject");
	cv::imshow("Colour Backproject",
				result);
}

void compareTest(){
	// Read reference image
	cv::Mat reference= cv::imread("../images/abelFace1.jpg");
	// Read compared images
	cv::Mat image1= cv::imread("../images/abelFace1.jpg");
	cv::Mat image2= cv::imread("../images/abelFace2.jpg");
	cv::Mat image3= cv::imread("../images/abel.jpg");
	cv::Mat image4= cv::imread("../images/boldt.jpg");
	cv::Mat image5= cv::imread("../images/boldt.jpg");
	
	ImageComparator c;
	c.setReferenceImage(reference);
	
	cout << " reference image score " <<c.compare(image1) << endl;
	cout << " image2 score " <<c.compare(image2) << endl;
	cout << " image3 score " <<c.compare(image3) << endl;
	cout << " image3 score " <<c.compare(image4) << endl;
	cout << " image3 score " <<c.compare(image5) << endl;
}

int main(){

	//~ histogramTest();
	//~ histogramImageTest();
	thresholdTest();
	//~ invertTest();
	//~ stretchTest();
	//~ equalizeTest();
	//~ finderTest();
	//~ hueTest();
	//~ compareTest();
	cv::waitKey();
	return 0;
}
