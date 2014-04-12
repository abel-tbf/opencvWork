#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

void salt(cv::Mat &image, int n){
	for(int k = 0; k < n; k++){
		// rand() is the MFC random number generator
		// try qrand() with Qt
		int i = rand() % image.cols;
		int j = rand() % image.rows;
		if(image.channels() == 1){ // gray-level image
			image.at<uchar>(j,i)=255;
		}else if(image.channels() == 3){ // color image
			image.at<cv::Vec3b>(j,i)[0]=255;
			image.at<cv::Vec3b>(j,i)[1]=255;
			image.at<cv::Vec3b>(j,i)[2]=255;
		}
	}
}

void colorReduce(cv::Mat &image, int div=64){
	int n1 = image.rows; // number of lines
	// total numbe of elements per line
	int nc=image.cols * image.channels();
	for(int j = 0; j < n1; j++){
		// get the addres of row j
		uchar * data = image.ptr<uchar>(j);
		for(int i = 0; i < nc; i++){
			// process each pixel ------------------
			data[i] = data[i] / div * div/2;
			// end of pixel processing--------------
		} // end of line
	}
}

void sharpen (const cv::Mat &image, cv::Mat &result){
	// allocate if necessary
	result.create(image.size(), image.type());
	for(int j = 1; j < image.rows - 1; j++){ // for all rows
		// (except first and last)
		const uchar* previous = 
			image.ptr<const uchar>(j-1); // previous row
		const uchar* current = 
			image.ptr<const uchar>(j); // current row
		const uchar* next=
			image.ptr<const uchar>(j+1); // next row
			
		uchar* output = result.ptr<uchar>(j); // output row
		
		for(int i = 1; i < image.cols - 1; i++){
			
			*output++ = cv::saturate_cast<uchar>(
							5*current[i] - current[i - 1]
							- current[i + 1] - previous[i] - next[i]);
		}
	}
	// Set the unprocess pixels to 0
	result.row(0).setTo(cv::Scalar(0));
	result.row(result.rows - 1).setTo(cv::Scalar(0));
	result.col(0).setTo(cv::Scalar(0));
	result.col(result.cols - 1).setTo(cv::Scalar(0));
}

void sharpen2D(const cv::Mat &image, cv::Mat &result){
	// Construct kernel (all entries initialized to 0)
	cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
	// assigns kernel values
	kernel.at<float>(1,1) = 5.0;
	kernel.at<float>(0,1) = -1.0;
	kernel.at<float>(2,1) = -1.0;
	kernel.at<float>(1,0) = -1.0;
	kernel.at<float>(1,2) = -1.0;
	
	// filter the image
	cv::filter2D(image,result,image.depth(),kernel);
}

int main(){
	//~ // open the image
	//~ cv::Mat image= cv::imread("../images/abel.jpg");
	cv::Mat image= cv::imread("../images/boldt.jpg");

	cv::Mat result;
	// call function to add noise
	//~ salt(image,3000);
	//~ colorReduce(image);
	//~ sharpen(image,result);
	sharpen2D(image,result);
	// display image
	cv::namedWindow("Image");
	//~ cv::imshow("Image",image);
	cv::imshow("Image",result);
	// wait for 5000 ms
	cv::waitKey(5000);
	return 1;
}
