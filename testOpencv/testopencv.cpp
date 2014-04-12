#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

int main(){
	// read image
	//~ cv::Mat image = cv::imread("../images/abel.jpg");
	//~ cv::Mat image = cv::imread("../images/puppy.jpg");
	cv::Mat image = cv::imread("../images/boldt.jpg");

	// create image window named "My image"
	cv::namedWindow("Mi image");
	// show the image in window
	cv::imshow("My image", image);
	// wait for 5000 ms
	cv::waitKey(5000);
	return 1;
}
