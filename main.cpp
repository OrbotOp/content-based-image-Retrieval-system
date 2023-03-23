/*author: HRITHIK KANOJE, RUSHI SHAH
Class: CS5330 Pattern Recog & Computer Vision
Prof: Bruce Maxwell
Project 2: Content-Based Image Retrievel
*/

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "functions.h"


#define BINS 16                                        //global defining bins=16
#define BASELINE_TARGET_IMAGE "olympus/pic.1016.jpg"          //target image for Task 1
#define COLOR_HIST_TARGET_IMAGE "olympus/pic.0164.jpg"        //target image for Task 2
#define MULTI_HIST_TARGET_IMAGE "olympus/pic.0274.jpg"        //target image for Task 3
#define TEXTURE_COLOR_HIST_TARGET_IMAGE "olympus/pic.0535.jpg"//target image for Task 4


using namespace cv;
using namespace std;



//sum-square diffrence as distance metric
float SSD(const cv::Mat &src, const cv::Mat &dst){
	int center_row = dst.rows /2;
	int center_col = dst.cols /2;
	float sums[]={0.f,0.f,0.f};

	for(int i = center_row -4; i<= center_row +4; i++){
		for(int j = center_col -4; j<= (center_col +4); j++){
			for(int k=0; k< dst.channels(); k++){
				int diff = dst.at<cv::Vec3b>(i,j)[k]-src.at<cv::Vec3b>(i,j)[k];
				sums[k] += float (diff*diff);

			}
		}
	}

	return sums[0]+sums[1]+sums[2];
}


//intesection as distance metric
float ID(const cv::Mat &m1, const cv::Mat &m2, bool is_3d){
	float intersection =0.f;
	if(is_3d){
		for(int i=0;i<m1.size[0];i++){
			for(int j=0;j<m1.size[1];j++){
				for(int k =0;k<m1.size[2];k++){
					intersection += std::min(m1.at<float>(i,j,k),m2.at<float>(i,j,k));
				}
			}
		}
	}
	else{
		for(int i=0;i<m1.rows;i++){
			for(int j=0;j<m1.cols;j++){
				intersection += std::min(m1.at<float>(i,j),m2.at<float>(i,j));
			}
		}
	}
	return intersection;
}

//normalized histogram
int hist_normal(cv::Mat &src, cv::Mat &dst, bool is_3d){
	long total = 0l;
	if(is_3d){
		for(int i=0; i<src.size[0];i++){
			for(int j=0; j<src.size[1];j++){
				for(int k=0;k<src.size[2];k++){
					total +=src.at<int>(i,j,k);
				}
			}
		}
		if (total ==0l)
			return(0);
		for(int i=0;i<dst.size[0];i++){
			for(int j=0;j<dst.size[1];j++){
				for(int k=0;k<src.size[2];k++){
					float per = (float)src.at<int>(i,j,k) / total;
					dst.at<float>(i,j,k) = per;
				}
			}
		}
	}
	else{
		for(int i=0;i<src.rows;i++){
			for(int j=0;j<src.cols;j++){
				total += src.at<int>(i,j);
			}
		}
		if(total ==0)
			return(0);
		for(int i=0;i<dst.rows;i++){
			for(int j=0;j<dst.cols;j++){
				float per = (float)src.at<int>(i,j) / total;
				dst.at<float>(i,j) = per;
			}
		}
	}
	return (0);
}

//rg_chrom histogram
int rg_chrom_hist(const cv::Mat &src, cv::Mat &dst){
	for(int i=0;i<src.rows; i++){
		for(int j=0;j<src.cols;j++){
			float p_sum = (float)src.at<cv::Vec3b>(i,j)[0] + (float)src.at<cv::Vec3b>(i,j)[1] + (float)src.at<cv::Vec3b>(i,j)[2] +1;
			int r_deno = src.at<cv::Vec3b>(i,j)[2] * BINS;
			int g_deno = src.at<cv::Vec3b>(i,j)[1] * BINS;
			float r = (float)r_deno/p_sum;
			float g = (float)g_deno/p_sum;
			dst.at<int>((int)g,(int)r) += 1;
		}
	}
	return(0);
}

//half-RGB histogram
int half_rgb_hist(const cv::Mat &src, cv::Mat &top, cv::Mat &bottom){
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			int ind[] ={0,0,0};
			for(int k=0;k<src.channels();k++){
				int deno = src.at<cv::Vec3b>(i,j)[k] * BINS;
				ind[k] = deno /256;
			}
			if (i<src.rows/2){
				top.at<int>(ind[0],ind[1],ind[2]) += 1;
			}
			else{
				bottom.at<int>(ind[0],ind[1],ind[2]) +=1;
			}
		}
	}
	return(0);
}

//seprable sobel
int spe_sobel(cv::Mat &src, cv::Mat &dst, const float hor_kernel[3],const float ver_kernel[3]){
	cv::Mat con(src.rows,src.cols, CV_16SC3);

	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			con.at<short>(i,j) = (short) (src.at<uchar>(i,j));
			dst.at<short>(i,j) = (short) (src.at<uchar>(i,j));
		}
	}

	//horizontal pass

	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols-3;j++){
			float b_total=0.0,g_total=0.0,r_total=0.0;
			for(int k=0;k<3;k++){
				b_total +=src.at<cv::Vec3b>(i,j+k)[0] * hor_kernel[k];
				g_total +=src.at<cv::Vec3b>(i,j+k)[1] * hor_kernel[k];
				r_total +=src.at<cv::Vec3b>(i,j+k)[2] * hor_kernel[k]; 
			}
			con.at<cv::Vec3s>(i,j+1) = cv::Vec3s(b_total,g_total,r_total);
		}
	}

	//vertical pass

	for(int i=0; i<=src.rows -3;i++){
		for(int j=0;j<=src.cols;j++){
			float b_total=0.0, g_total=0.0, r_total=0.0;
			for(int k=0;k<3;k++){
				b_total += con.at<cv::Vec3b>(i+k, j)[0] * ver_kernel[k];
				g_total += con.at<cv::Vec3b>(i+k, j)[1] * ver_kernel[k];
				r_total += con.at<cv::Vec3b>(i+k, j)[2] * ver_kernel[k];
			}
			dst.at<cv::Vec3s>(i+1,j) = cv::Vec3s(b_total,g_total,r_total);
		}
	}
	return(0);
}

//sobel X
int sobelX3x3(cv::Mat &src, cv::Mat &dst){
	float hor_kernel[3] = {1.0f,0.0f,-1.0f};
	float ver_kernel[3] = {0.25f,0.5f,0.25f};
	spe_sobel(src,dst,hor_kernel,ver_kernel);
	return(0);
}

//sobel Y
int sobelY3x3(cv::Mat &src, cv::Mat &dst){
	float hor_kernel[3] = {0.25f,0.5f,0.25f};
	float ver_kernel[3] = {1.0f,0.0f,-1.0f};
	spe_sobel(src,dst,hor_kernel,ver_kernel);
	return(0);
}

//magnitude from sobel X and sobel Y
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){
	for(int i=0;i<sx.rows;i++){
		for(int j=0;j<sx.cols;j++){
			cv::Vec3s sx_pxl = sx.at<cv::Vec3s>(i,j);
			cv::Vec3s sy_pxl = sy.at<cv::Vec3s>(i,j);
			dst.at<cv::Vec3b>(i,j) = cv::Vec3b(
				                               sqrt(sx_pxl[0] * sx_pxl[0]+sy_pxl[0]*sy_pxl[0]),
				                               sqrt(sx_pxl[1] * sx_pxl[1]+sy_pxl[1]*sy_pxl[1]),
				                               sqrt(sx_pxl[2] * sx_pxl[2]+sy_pxl[2]*sy_pxl[2]));
		}
	}
	return(0);
}

//texture histogram
int texture_hist(const cv::Mat &src, int *bins, float *normal_bins){
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			for(int k=0;k<src.channels();k++){
				int deno = src.at<cv::Vec3b>(i,j)[k] * BINS;
				bins[deno/256] +=1;
			}
		}
	}
	int total = 1;
	for(int i=0;i<BINS;i++){
		total += bins[i];
	}
	for (int i=0;i<BINS;i++){
		normal_bins[i] = (float)bins[i] /total;
	}
	return(0);
}



int readFiles(char *img_dir, std::vector<std::string> &files) //From the prof's sample code for reading files
{
	char dirname[256];
	char buffer[256];
	DIR *dirp;
	struct dirent *dp;

	strcpy(dirname, img_dir);
	printf("Processing directory %s \n",dirname);
    
    //open the directory
	dirp = opendir(dirname);
	if (dirp == NULL)
	{
		printf("Cannot open the directory %s \n",dirname);
		exit(-1);
	}

	//loop over all the files in the the image file listing
	while((dp= readdir(dirp)) != NULL)
	{
        //check if the file is image 
		if(strstr(dp->d_name, ".jpg") ||
		   strstr(dp->d_name, ".png") ||
		   strstr(dp->d_name, ".ppm") ||
		   strstr(dp->d_name, ".tif")){

			//buid the overall filename
			strcpy(buffer, dirname);
		    strcat(buffer, "/");
		    strcat(buffer, dp->d_name);

		    files.emplace_back(buffer);

		}
	}
	return (0);
}


//creating a pipeline for performing all the task from feature extraction, getting distance distance between target image and database image and return top three matches
int process_pipeline(const std::string &src_dir,const std::vector<std::string> &files,std::vector<std::string> &top_number, int task){

	cv::Mat src = cv::imread(src_dir);
	cv::Mat dst(BINS, BINS, CV_32SC1, cv::Scalar(0));
	cv::Mat normal_dst(BINS, BINS, CV_32F, cv::Scalar(0));

	int size[]={BINS, BINS, BINS};
	cv::Mat top(3, size, CV_32SC1, cv::Scalar(0));
	cv::Mat bottom(3, size, CV_32SC1, cv::Scalar(0));
	cv::Mat normal_top(3, size, CV_32F, cv::Scalar(0));
	cv::Mat normal_bottom(3, size,CV_32F, cv::Scalar(0));

	cv::Mat greyscale_magnitude(src.rows,src.cols, CV_8UC1, cv::Scalar(0));
	int bins[BINS] = {0};
	int vertical_bins[BINS] = {0};
	float normal_bins[BINS] = {0.f};
	float normal_vertical_bins[BINS] = {0.f};
	

	// all the filename as keys and distance score as values in a map
	std::vector<std::pair<std::string, float>> map;
	for (const std::string &file: files){
		map.emplace_back(std::make_pair(file, 0.));
	}

	//for the diffrent tasks, diffrent functions are called
	if(task ==2){
		rg_chrom_hist(src,dst);
		hist_normal(dst,normal_dst, false);
	}
	else if (task ==3){
		half_rgb_hist(src, top, bottom);
		hist_normal(top, normal_top, true);
		hist_normal(bottom, normal_bottom,true);
	}
	else if (task ==4){
		cv::Mat x_sobel(src.rows, src.cols, CV_16SC3, cv::Scalar(0));
		cv::Mat y_sobel(src.rows, src.cols, CV_16SC3, cv::Scalar(0));
		cv::Mat mag(src.rows, src.cols, CV_8UC3, cv::Scalar(0));
		rg_chrom_hist(src,dst);
		hist_normal(dst,normal_dst,false);
		sobelX3x3(src,x_sobel);
		sobelY3x3(src,y_sobel);
		magnitude(x_sobel,y_sobel,mag);
		cv::cvtColor(mag,greyscale_magnitude, cv::COLOR_BGR2GRAY);
		texture_hist(greyscale_magnitude,bins,normal_bins);
	}
	
	

	for(std::pair<std::string, float> &m: map){
		cv::Mat image = cv::imread(m.first);

		if (task ==1){
			m.second = SSD(src,image);
		}
		else if( task ==2){
			cv::Mat image_hist = cv::Mat(BINS,BINS, CV_32SC1, cv::Scalar(0));
			cv::Mat image_normal_dst = cv::Mat(BINS, BINS ,CV_32F,cv::Scalar(0));
			rg_chrom_hist(image,image_hist);
			hist_normal(image_hist, image_normal_dst, false);
			m.second = ID(normal_dst,image_normal_dst,false);
		}
		else if(task ==3){
			cv::Mat image_top = cv::Mat(3,size,CV_32SC1,cv::Scalar(0));
			cv::Mat image_bottom = cv::Mat(3,size,CV_32SC1,cv::Scalar(0));
		    cv::Mat image_normal_top = cv::Mat(3,size,CV_32F,cv::Scalar(0));
		    cv::Mat image_normal_bottom =cv::Mat(3,size,CV_32F,cv::Scalar(0));
		    half_rgb_hist(image,image_top,image_bottom);
		    hist_normal(image_top,image_normal_top, true);
		    hist_normal(image_bottom,image_normal_bottom, true);
		    m.second = ID(normal_top,image_normal_top, true) * 0.5f + ID(normal_bottom,image_normal_bottom, true) *0.5f;
        }
        else if(task ==4){
        	cv::Mat image_dst = cv::Mat(BINS,BINS,CV_32SC1,cv::Scalar(0));
        	cv::Mat image_normal_dst = cv::Mat(BINS,BINS, CV_32F, cv::Scalar(0));
            cv::Mat image_greyscale_mag(image.rows,image.cols, CV_8UC1, cv::Scalar(0));
            cv::Mat image_x_sobel(image.rows,image.cols, CV_16SC3, cv::Scalar(0));
            cv::Mat image_y_sobel(image.rows,image.cols, CV_16SC3, cv::Scalar(0));
            cv::Mat mag(image.rows,image.cols, CV_8UC3, cv::Scalar(0));


        	rg_chrom_hist(image,image_dst);
        	hist_normal(image_dst, image_normal_dst, false);
        	sobelX3x3(image,image_x_sobel);
        	sobelY3x3(image,image_y_sobel);
        	magnitude(image_x_sobel,image_y_sobel,mag);
        	cv::cvtColor(mag,image_greyscale_mag, cv::COLOR_BGR2GRAY);
            
            int image_bins[BINS] = {0};
            float image_normal_bins[BINS] = {0.f};
            texture_hist(image_greyscale_mag, image_bins,image_normal_bins);

            float text_score = 0.f;
            for(int i=0; i< BINS;i++){
            	text_score += std::min(normal_bins[i], image_normal_bins[i]);
            }
            m.second = text_score * 0.5f + ID(normal_dst,image_normal_dst,false) * 0.5f;
        }
       
       
	}

	// sorting the map by image name and distance by value

	if (task == 1){
		sort(map.begin(),map.end(),[=](std::pair<std::string, float> &a, std::pair<std::string, float> &b){
			return a.second < b.second;
		});
	}
	else{
		sort(map.begin(),map.end(), [=](std::pair<std::string, float> &a, std::pair<std::string, float> &b){
			return a.second >b.second;
		});
	}

	if(task != 5){
		//assrt the first image
		assert(map.at(0).first == src_dir);
        
        //return top three images except itself
        top_number.push_back(map.at(1).first);
        top_number.push_back(map.at(2).first);
        top_number.push_back(map.at(3).first);
        std::cout << "Top 3 Matches "<<std::endl;
        for(int i=1;i<4;i++){
        	std::cout<<map.at(i).first<<std::endl;
        }
    }    
    else{
       	std::cout<<"TOP 10 Matches (TASK5)"<<std::endl;
       	for(int i=1;i<11;i++){
       		std::cout<<map.at(i).first<<std::endl;
        	}
        	top_number.push_back(map.at(1).first);
        	top_number.push_back(map.at(2).first);
        	top_number.push_back(map.at(3).first);
       }

return(0);
}

int main(int argc, char *argv[])
{
	cv::Mat src;
	int cols = 640;    //All the images of directory is of same 640 x 512 pixels
	int rows = 512;
	std::vector<std::string> files;
	std::vector<std::string> top_number;

	if (argc <3)
	{
		printf("usage:%s <directory path> <Task>\n",argv[0]);
		exit(-1);
	}
	readFiles(argv[1], files);

	char *taskArg = argv[2];
	if (strlen(taskArg) > 1)
	{
		printf("Selected Task should be from 1 to 4");
		exit(-1);
	}

	int task = taskArg[0]-'0';
	switch (task)
	{
      case 1:{
      	process_pipeline(BASELINE_TARGET_IMAGE, files, top_number, task);
      	break;
      }
      case 2:{
      	process_pipeline(COLOR_HIST_TARGET_IMAGE, files,top_number, task);
        break;
      }
      case 3:{
      	process_pipeline(MULTI_HIST_TARGET_IMAGE, files,top_number, task);
      	break;
      }
      case 4:{
        process_pipeline(TEXTURE_COLOR_HIST_TARGET_IMAGE, files, top_number, task);
        break;
       }
      default:{
      	printf("Task should be one of the number from 1 to 4");
      	exit(-1);
      } 
	}
	



	return 0;
}