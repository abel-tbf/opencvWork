#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
//~ #include<iostream>

//~ using namespace std;
class ColorDetector {
	private:
		// minimum acceptable distance
		int minDist;
		
		// target color
		cv::Vec3b target;
		
		// image containing resulting binary map
		cv::Mat result;
		
		// computes de distance from target color
		int getDistance(const cv::Vec3b& color) const{
			//~ return abs(color[0] - target[0]) +
					//~ abs(color[1] - target[1]) +
					//~ abs(color[2] - target[2]);
			return static_cast<int>(
					cv::norm<int,3>(cv::Vec3i(color[0] - target[0],
											color[1] - target[1],
											color[2] - target[2])));
		}
	
	public:
		// empty constructor
		ColorDetector() : minDist(100){
			// default parameter initialization here
			target[0] = target[1] = target[2] = 0;
		}
		
		// Sets the color distance threshold.
		// Threshold must be positive,
		// otherwise distance threshold is set to 0.
		void setColorDistanceThreshold(int distance) {
			if (distance < 0)
				distance = 0;
			minDist = distance;
		}
		
		// Gets the color distance threshold
		int getColorDistanceThreshold() const {
			return minDist;
		}
		
		// Sets the color to be detected
		void setTargetColor(unsigned char red, unsigned char green, 
							unsigned char blue){
		
			// BGR order
			target[2] = red;
			target[1] = green;
			target[0] = blue;
		}
		
		// Sets the color to be detected
		void setTargetColor(cv::Vec3b color){
			target = color;
		}
		
		// Gets the color to be detected
		cv::Vec3b getTargetColor() const{
			return target;
		}
		
		cv::Mat process(const cv::Mat &image){
			// re-allocate binary map if necessary
			// same size as input image, but 1-channel
			result.create(image.rows,image.cols,CV_8U);
			
			// get the iterators 
			cv::Mat_<cv::Vec3b>::const_iterator it = image.begin<cv::Vec3b>();
			cv::Mat_<cv::Vec3b>::const_iterator itend = image.end<cv::Vec3b>();
			cv::Mat_<uchar>::iterator itout = result.begin<uchar>();
			
			// for each pixel
			for(; it!= itend; ++it, ++itout){
				// process each pixel --------------------
				// compute distance from target color
				if(getDistance(*it)<minDist){
					*itout = 255;
				}else{
					*itout = 0;
				}
				// end of pixel processing ----------------
			}
			
			return result;
		}
};

int main(){
	// 1. Create image processor object
	ColorDetector cdetect;
	
	// 2. Read input image
	//~ cv::Mat image = cv::imread("../images/abel.jpg");
	cv::Mat image = cv::imread("../images/boldt.jpg");
	
	if(!image.data){
		//~ cout << "no data" << endl;
		return 0;
	}
	
	// 3. Set input parameters
	cdetect.setTargetColor(130,190,230);
	
	cv::namedWindow("result");
	
	// 4. Process the image and display the result
	cv::imshow("result",cdetect.process(image));
	
	cv::waitKey();
	return 0;
}
