/*functions include file
 Author: HRITHIK KANOJE , RUSHI SHAH
 Class: CS5330 Pattern Recog & Computer vision
 Prof: Bruce maxwell
 project 2:Content-based Image Retrieval
*/



//function for calculating sum-of-squared-diffrence
float SSD(const cv::Mat &src, const cv::Mat &dst);

//function for calculating histogram intersection
float ID(const cv::Mat &m1, const cv::Mat &m2, bool is_3d);

//function for normalized histogram
int hist_normal(cv::Mat &src, cv::Mat &dst, bool is_3d);

//function for rg chrom. histogram
int rg_chrom_hist(const cv::Mat &src, cv::Mat &dst);

//function for Half-RGB histogram
int half_rgb_hist(const cv::Mat &src, cv::Mat &top, cv::Mat &bottom);

//function for sperable sobel
int spe_sobel(cv::Mat &src, cv::Mat &dst, const float hor_kernel[3],const float ver_kernel[3]);

//function for 3x3 sobel X
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

//function for 3x3 sobel Y
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

//function for magnitude
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

