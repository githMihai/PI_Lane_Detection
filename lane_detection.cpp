#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <time.h>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp" 
#include "opencv2/highgui.hpp"

#include <queue>

#include <unistd.h>

#define PI 3.14159265
// #define keyboard 0

using namespace cv;
using namespace std;

void HoughTransform(Mat src, Mat gradient)
{
	Mat hough = Mat::zeros(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.data[i*src.step[0] + j] == 255)
			{
				float tetha = gradient.data[i*src.step[0] + j];
				int p = i*cos(tetha) + j*sin(tetha);
				hough.data[(int)(tetha*180./PI)*hough.step[0] + p]++;
			}
		}
	}
}


void smoothingFilter(Mat src, uchar* dst, int width, int height, int w)
{
	int half_w = w/2;
	for (int i = half_w; i < height-half_w; i++)
	{
		int iwidth = i*width;
		int aux = 0;
		for (int j = half_w; j < width-half_w; j++)
		{
			aux =	(src.at<uchar>(i-1,j+1) + src.at<uchar>(i-1,j) + src.at<uchar>(i+1,j+1) +
					src.at<uchar>(i,j-1) + src.at<uchar>(i,j) + src.at<uchar>(i,j+1) +
					src.at<uchar>(i+1,j-1) + src.at<uchar>(i+1,j) + src.at<uchar>(i+1,j+1))/9;
			dst[iwidth + j] = aux;
		}
	}
}


/**
 * ################################################################
 * 						GAUSSIAN NOISE REDUCTIOM
 * ################################################################
**/ 
void gaussian_noise_Y(uchar* src, uchar* dst, int width, int height, int w)
{
	float sigma = float(w/6.0);

	static float* Gx = new float[w];
	static bool modified = false;

	if (!modified)
	{
		for (int i = 0 ; i < w; i++)
		{
				Gx[i] = (1/(pow(2*PI, 0.5)*sigma))*(exp( -( pow(i - w/2,2)/(2*pow(sigma,2)) )));
		}
		modified = true;
		// cout << "Gx: " << Gx[0] << endl;
	}

	for (int i = w/2; i < height - w/2; i++)
	{
		int iwidth = i*width;
		for (int j = w/2; j < width - w/2; j++)
		{
			float aux = 0.;
			for (int y = 0; y < w; y++)
			{
				// cout << "src: " << (float)(src[iwidth + j-w/2+y]) << " Gx: " << Gx[y] << endl;
				aux += (float)(src[iwidth + j-w/2+y])*Gx[y];
			}
			dst[iwidth + j] = aux;
			// cout << "aux = " << aux << endl;

			if (i > w)
			{
				int iwidth_prev = (i-w+1)*width;
				aux = 0.;
				for (int y = 0; y < w; y++)
				{
					aux += dst[j + iwidth_prev-w/2+y]*Gx[y];
				}
				dst[iwidth_prev + j] = aux;
			}
		}
	}
}



/**
 * ################################################################
 * 						GET NEIGHBOURS NUMBER
 * ################################################################
**/ 
int getNeighboursNo(uchar *src, int i, int j, int width)
{
	int iwidth = i*width;
	int neighbours = 0;
	if (src[iwidth - width + j-1] == 255) { neighbours++; }
	if (src[iwidth - width + j] == 255) { neighbours++; }
	if (src[iwidth - width + j+1] == 255) { neighbours++; }

	if (src[iwidth + j-1] == 255) { neighbours++; }
	if (src[iwidth + j+1] == 255) { neighbours++; }

	if (src[iwidth + width + j-1] == 255) { neighbours++; }
	if (src[iwidth + width + j] == 255) { neighbours++; }
	if (src[iwidth + width + j+1] == 255) { neighbours++; }

	return neighbours;
}


/**
 * ################################################################
 * 					GET DIRECTION OF A GIVEN ANGLE
 * ################################################################
**/ 
int getDirection(float angle)
{
	if (angle >= 3*PI/8.0 && angle <= 5*PI/8.0)
	{
		return 0;
	}else
	if (angle >= PI/8.0 && angle <= 3*PI/8.0)
	{
		return 1;
	}else
	if (angle >= -PI/8.0 && angle <= PI/8.0)
	{
		return 2;
	}else
	if (angle >= -3*PI/8.0 && angle <= -PI/8.0)
	{
		return 3;
	}else
	if (angle >= -5*PI/8.0 && angle <= -3*PI/8.0)
	{
		return 0;
	}else
	if (angle >= -7*PI/8.0 && angle <= -5*PI/8.0)
	{
		return 1;
	}else
	if (angle >= 5*PI/8.0 && angle <= 7*PI/8.0)
	{
		return 3;
	}
	else
		return 2;
	return 0;
}


/**
 * ################################################################
 * 			SELECT THE TWO LINES THAT CONTAINS THE MOST POINTS
 * ################################################################
 * 
 * 	Get the coordinates [tetha, p] from Hough space with the highest value
 * (where the most lines from image space intersect) and this coordinates
 * will be the parameters for the line in image space: 
 * 				x*cos(tetha) + y*sin(tetha) = p
 * where (x,y) is eny point from image space
**/
vector<Point2d> houghPoints(uchar* src, Mat hough, int width, int height, int min_line, int max_line, bool limited_lines=true)
{
	Mat mark = Mat::zeros(height, width, CV_8UC3);

	bool left = false;
	bool right = false;
	vector<Point2d> points;

	// if (!left || !right || !limited_lines)
	{
		
		Point2d p0_l;
		Point2d p1_l;
		Point2d p0_r;
		Point2d p1_r;

		int max_l = 0;
		int max_r = 0;

		/**
		 * For every angle from 0 to 180 compute
		 **/ 
		for (int i = 0; i < 180; i++)
		{
			/**
			 * If the two lines were not found
			 **/
			// if (!left || !right || !limited_lines)
			{
				/**
				 * For every p from Hough's matrix
				 **/ 
				for (int j = 0; j < hough.step[0]; j++)
				{
					/**
					 * Find 2 points that cross the image from top to bottom
					 **/
					float tetha = (float)(i)*PI/180.;

					Point2d p0 = Point2d(j, min_line);
					Point2d p1 = Point2d((int)((j-max_line*sin(tetha))/cos(tetha)), max_line-1);
					/** Left line **/
					if (p0.x < width/2 - 10 && p1.x < width/2 - 10 && 
						p0.x > 10 && p1.x > 10 && !left || !limited_lines)
					{
						/**
						 * If the computed line contain more then 50 points keep it
						 **/
						if (hough.data[hough.step[0]*i + j] >= max_l && /*p0.x != p1.x &&*/
							p0.x > 2 && p0.x < width-2 && p1.x > 2 && p1.x < width-2)
						{
							// line(mark,p0, p1,(0,0,255),4);
							// if (!left)
							// {
							// 	// cout << "p0.x,p1.x:  " << p0.x << ", " << p1.x << endl;
							// 	points.push_back(p0);
							// 	points.push_back(p1);
							// 	left = true;
							// }
							p0_l = p0;
							p1_l = p1;
							// line(mark,p0_l, p1_l,(0,0,255),4);
							max_l = hough.data[hough.step[0]*i + j];
						}
					}
					/** Right line **/
					else if (p0.x > width/2 + 10 && p1.x > width/2 + 10 
							&& p0.x < width-10 && p1.x < width-10 /*&& !right || !limited_lines*/)
					{
						/**
						 * If the computed line contain more then 20 points keep it
						 **/
						if (hough.data[hough.step[0]*i + j] >= max_r /*&& p0.x != p1.x &&
							p0.x > 2 && p0.x < width-2 && p1.x > 2 && p1.x < width-2*/)
						{
							// line(mark,p0, p1,(0,0,255),4);
							// if (!right)
							// {
							// 	points.push_back(p0);
							// 	points.push_back(p1);
							// 	right = true;
							// }
							p0_r = p0;
							p1_r = p1;
							// line(mark,p0_r, p1_r,(0,0,255),4);
							max_r = hough.data[hough.step[0]*i + j];
						}
					}
				}
			}
		}
		line(mark,p0_l, p1_l, Scalar(0,0,255),4);
		line(mark,p0_r, p1_r, Scalar(0,0,255),4);
		points.push_back(p0_l);
		points.push_back(p1_l);
		points.push_back(p0_r);
		points.push_back(p1_r);
	}

	// imshow("Hough", hough);
	// cout << "pH: " << points.at(0) << ", " << points.at(1) << ", " << points.at(2) << ", " << points.at(3) << endl;
	imshow("Mark", mark);
	// waitKey();
	// return mark; 
	return points;       
}

/**
 * ################################################################
 * 				FIND THE CENTER LINE OF TWO LINE
 * ################################################################
**/ 
vector<Point2d> findCenterLine(vector<Point2d> points)
{
	Point2d center_up = Point2d((points[0].x + points[2].x)/2, (points[0].y + points[2].y)/2);
	Point2d center_down = Point2d((points[1].x + points[3].x)/2, (points[1].y + points[3].y)/2);
	points.push_back(center_up);
	points.push_back(center_down);
	return points;
}

/**
 * ################################################################
 *	 			DETECT THE EDGES OF THE INPUT IMAGE,
 *			COMPUTE THE GRADIENT FOR EVERY EDGE POINT,
 *						COMPUTE HOUGH MATRIX				
 * ################################################################
 * 
 *  This function applies the two sobel convolution operatos in an efficient
 * way using pointers to matrix elements and than compute the absolute value
 * and applies thresholding (threshold = 50).
 * 	Also, this function calculate the angle (gradient) for every edge point and
 * make the Hoigh matrix
**/
vector<Point2d> edgesDetection(uchar* src, uchar* dst, int width, int height, int min_line, int max_line, int threshold)
{
	Mat phi = Mat::zeros(height, width, CV_32FC1);
	vector<Point2d> points;

	int max = 0;

	/** Maximum size of the Hough matrix **/
	int diag  = sqrt(width*width + height*height);
	Mat hough = Mat::zeros(180, diag, CV_8UC1);

	/**
	 * For every element in the input matrix
	 **/
	for (int i = 1; i < height - 1; i++)
	{
		int iwidth = i * width;
		for (int j = 1; j < width - 1; j++)
		{
			/**
			 * Compute the two sobel gradients
			 **/
			int sX = 
						-src[iwidth - width + j - 1]	+	src[iwidth - width + j + 1] +
						-(src[iwidth + j - 1] << 1)		+	(src[iwidth + j - 1] << 1)	+
						-src[iwidth + width + j - 1]	+	src[iwidth + width + j + 1];

			int sY = 
						-src[iwidth - width + j - 1]	+	src[iwidth + width + j - 1] +
						-(src[iwidth - width + j] << 1)	+	(src[iwidth + width + j] << 1)	+
						-src[iwidth - width + j + 1]	+	src[iwidth + width + j + 1];

			/**
			 * Absolute value, thresholding and angle (gradient)
			 **/
			dst[iwidth + j] = (sqrt(sX*sX + sY*sY)) > 50 ? 255 : 0;
			phi.at<float>(i,j) = (atan2((float)sY, (float)sX));

			/**
			 * #########################################
			 * 					Hought
			 * #########################################
			**/
			if (dst[iwidth + j] == 255)
			{
				/**
				 * For every edge point compute the p value using line ecuation
				 * 			x*cos(tetha) + y*sin(tetha) = p
				 * where tetha is the gradient. The parameters tetha and p will
				 * determine a line in the [tetha,p] space (Hough space) and the
				 * calue of the hough Matrix in the point [tetha,p] is incremented
				 * for every lines intersection
				 **/
				float tetha = abs(phi.at<float>(i,j));
				float p = (float)(j)*cos(tetha) + (float)(i)*sin(tetha);
				// cout << "p: " << p << endl;
				hough.at<uchar>((int)(tetha*180./PI), int(p))++;
				if (hough.at<uchar>((int)(tetha*180./PI), int(p)) > max)
				{
					max = hough.at<uchar>((int)(tetha*180./PI), int(p));
				}
			}

			// if (i > 3)
			// {
			// 	int iwidth_prev = (i-2) * width;
			// 	int direct = getDirection(phi.data[iwidth + j]);

			// 	switch(direct)
			// 	{
			// 		case 0:	{ 	if (dst[iwidth_prev + j] <= dst[iwidth_prev-width + j] ||
			// 						dst[iwidth_prev + j] <= dst[iwidth_prev+width + j])
			// 						dst[iwidth_prev + j] = 0; 
			// 						// printf("direct 0\n");
			// 						break;
			// 						}
			// 		case 1:	{ 	if (dst[iwidth_prev + j] <= dst[iwidth_prev-width + j+1] ||
			// 						dst[iwidth_prev + j] <= dst[iwidth_prev+width + j-1])
			// 						dst[iwidth_prev + j] = 0; 
			// 						// printf("direct 1\n");
			// 						break;
			// 						}
			// 		case 2:	{ 	if (dst[iwidth_prev + j] <= dst[iwidth_prev + j-1] ||
			// 						dst[iwidth_prev + j] <= dst[iwidth_prev + j+1])
			// 						dst[iwidth_prev + j] = 0; 
			// 						// printf("direct 2\n");
			// 						break;
			// 						}
			// 		case 3:	{ 	if (dst[iwidth_prev + j] <= dst[iwidth_prev-width + j-1] ||
			// 						dst[iwidth_prev + j] <= dst[iwidth_prev+width + j+1])
			// 						dst[iwidth_prev + j] = 0; 
			// 						// printf("direct 3\n");
			// 						break;
			// 						}
			// 	}

			// 	if (dst[iwidth_prev + j] > threshold && (
			// 		// (dst[iwidth_prev - width + j-1 ] > threshold ||
			// 		dst[iwidth_prev - width + j] > threshold ||
			// 		// dst[iwidth_prev - width + j+1 ] > threshold ||
			// 		dst[iwidth_prev + j-1 ] > threshold ||
			// 		dst[iwidth_prev + j+1 ] > threshold ||
			// 		// dst[iwidth_prev + width + j-1 ] > threshold ||
			// 		dst[iwidth_prev + width + j ] > threshold ))
			// 		// dst[iwidth_prev + width + j+1 ] > threshold ))
			// 	{
			// 		if (iwidth_prev/width > 4 && iwidth_prev/width < height-4 &&
			// 			j > 4 && j < width-4)
			// 		{
			// 			dst[iwidth_prev + j] = 255;
			// 			points.push_back({iwidth_prev/width,j});
			// 		}
			// 		else
			// 		{
			// 			dst[iwidth_prev + j] = 0;
			// 		}
			// 	}
			// 	else
			// 	{
			// 		dst[iwidth_prev + j] = 0;
			// 	}
			// }
		}

	}


	/**
	 * ##############################################
	 * 					DILATION
	 * ##############################################
	**/
	// Mat dilation_matrix = Mat::zeros(height, width, CV_8UC3);
	// uchar* dilation = dilation_matrix.data;
	// for (int i = 7; i < height-7; i++)
	// {
	// 	int iwidth = i*width;
	// 	for (int j = 7; j < width-7; j++)
	// 	{
	// 		int value = 0;
	// 		if (dst[iwidth + j] == 255 && getNeighboursNo(dst, i, j, width) >=2)
	// 		{
	// 			value = 255;
	// 			for (int u = -4; u < 4; u++)
	// 			{
	// 				for (int v = -4; v < 4; v++)
	// 				{
	// 					dilation[iwidth - u*width + j-v] = value;
	// 				}
	// 			}
	// 		}
	// 		else
	// 		{
	// 			dilation[iwidth + j] = value;
	// 		}
			
	// 	}
	// }

	/**
	 * #########################################
	 * 					Hought
	 * #########################################
	**/
	// dst = dilation;

	// for (int i = 10; i < height-10; i++)
	// {
	// 	int iwidth = i * width;
	// 	for (int j = 10; j < width-10; j++)
	// 	{
	// 		if (dst[iwidth + j] == 255)
	// 		{
	// 			float tetha = abs(phi.at<float>(i,j));
	// 			// printf("tetha: %.2f\n", tetha*180./PI);
	// 			float p = j*cos(tetha) + i*sin(tetha);
	// 			// printf("p: %.2f\n", p);
	// 			hough.at<uchar>((int)(tetha*180./PI), int(p))++;
	// 			if (hough.at<uchar>((int)(tetha*180./PI), int(p)) > max)
	// 			{
	// 				max = hough.at<uchar>((int)(tetha*180./PI), int(p));
	// 			}
	// 			// cout << "(t,p) = " << (int)(tetha*180./PI) << " " << (int)p << endl;
	// 		}
	// 	}
	// 	// cout << hough << endl;
	// }
	// cout << "max: " << max << endl;

	// Mat mark = houghPoints(dst, hough, width, height);
	points = houghPoints(dst, hough, width, height, min_line, max_line, true);
	points = findCenterLine(points);
	// cout << "pE: " << points.at(0) << ", " << points.at(1) << ", " << points.at(2) << ", " << points.at(3) << endl;
	return points;
	// return mark;
	// imshow("m0", mark);
	// imshow("m1", mark1);
	// waitKey(0);
	// return points;
}


/**
 * ################################################################
 * 					COMPUTE HOMOGRAPHY MATRIX 
 * 								FOR 
 * 					PERSPECTIVE TRANSFORMATION
 * ################################################################
 * 
 * 	This function computes a matrix H that will wrap an image detemined
 * by a set of 4 points into another image determined by another 4 points.
 * 
 * 			⎡ u ⎤		⎡ H00  H01  H02 ⎤   ⎡ x ⎤
 * 			⎢ v ⎥   =   ⎢ H10  H11  H12 ⎥ * ⎢ y ⎥
 * 			⎣ w ⎦		⎣ H20  H21  H22 ⎦	⎣ 1 ⎦
 * 
 * 
 *   		⎡ −x1   −y1   −1     0     0    0   x1*x′1   y1*x′1   x′1 ⎤		⎡ h1 ⎤
 *     		⎢   0     0    0   −x1   −y1   −1   x1*y′1   y1*y′1   y′1 ⎥		⎢ h2 ⎥
 *	     	⎢ −x2   −y2   −1     0     0    0   x2*x′2   y2*x′2   x′2 ⎥		⎢ h3 ⎥
 * P*H  = 	⎢   0     0    0   −x2   −y2   −1   x2*y′2   y2*y′2   y′2 ⎥	 * 	⎢ h4 ⎥  =  0
 *   		⎢ −x3    −y3  −1     0     0    0   x3*x′3   y3*x′3   x′3 ⎥		⎢ h5 ⎥
 *   		⎢   0      0   0   −x3   −y3   −1   x3*y′3   y3*y′3   y′3 ⎥		⎢ h6 ⎥
 *   		⎢ −x4    −y4  −1     0     0    0   x4*x′4   y4*x′4   x′4 ⎥		⎢ h7 ⎥
 *   		⎣   0      0   0   −x4   −y4   −1   x4*y′4   y4*y′4   y′4 ⎦		⎢ h8 ⎥
 *																		    ⎣ h9 ⎦
 * 
 * We need to find the values: h1..h9 using SVD method
 **/
Mat getHomographyMatrix(Point2f *src_vertices, Point2f *dst_vertices)
{
	Mat A = Mat::zeros(8, 9, CV_32FC1);
	Mat H = Mat::zeros(9, 1, CV_32FC1);
	Mat Hf = Mat::zeros(3, 3, CV_32FC1);

	for (int i = 0; i < 8; i+=2)
	{
		// cout << i/2 << endl;
		A.at<float>(i, 0) = src_vertices[i/2].x;
		A.at<float>(i, 1) = src_vertices[i/2].y;
		A.at<float>(i, 2) = 1.;
		A.at<float>(i, 6) = -dst_vertices[i/2].x*src_vertices[i/2].x;
		A.at<float>(i, 7) = -dst_vertices[i/2].x*src_vertices[i/2].y;
		A.at<float>(i, 8) = -dst_vertices[i/2].x;

		A.at<float>(i+1, 3) = src_vertices[i/2].x;
		A.at<float>(i+1, 4) = src_vertices[i/2].y;
		A.at<float>(i+1, 5) = 1.;
		A.at<float>(i+1, 6) = -dst_vertices[i/2].y*src_vertices[i/2].x;
		A.at<float>(i+1, 7) = -dst_vertices[i/2].y*src_vertices[i/2].y;
		A.at<float>(i+1, 8) = -dst_vertices[i/2].y;
	}

	// cout << A << endl;

	SVD::solveZ(A, H);

	Hf.at<float>(0,0) = H.at<float>(0);
	Hf.at<float>(0,1) = H.at<float>(1);
	Hf.at<float>(0,2) = H.at<float>(2);
	Hf.at<float>(1,0) = H.at<float>(3);
	Hf.at<float>(1,1) = H.at<float>(4);
	Hf.at<float>(1,2) = H.at<float>(5);
	Hf.at<float>(2,0) = H.at<float>(6);
	Hf.at<float>(2,1) = H.at<float>(7);
	Hf.at<float>(2,2) = H.at<float>(8);

	//  cout << Hf << endl;

	return Hf;
}

void commented()
{
	/**
	 * ################################################################
	 * 					PERSPECTIVE WRAPING
	 * ################################################################
	 * 
	 * 	This function translate a selected plane from an image into a destination plane
	 * using a Homography matrix. This is done by multipling every point with
	 * the Homography matrix to optain the new point
	 **/
	// void perspectiveWrap(Mat perspective_image, uchar *orthogonal, float* homographyMatrix, int x_min, int x_max, int y_min, int y_max, int width, int height)
	// {
	// 	int i = y_min;
	// 	int prev_x = 0;
	// 	int prev_y = 0;
	// 	int x;
	// 	int y;
	// 	/**
	// 	 * For every pixel in the selected region aply the transformation
	// 	 **/
	// 	while (i < y_max)
	// 	{
	// 		int j = x_min;
	// 		while (j < x_max)
	// 		{
	// 			{
	// 				float u = 	homographyMatrix[0]*(float)j + 
	// 							homographyMatrix[1]*(float)i + 
	// 							homographyMatrix[2]*1.;
	// 				float v = 	homographyMatrix[3]*(float)j + 
	// 							homographyMatrix[4]*(float)i + 
	// 							homographyMatrix[5]*1.;
	// 				float w = 	homographyMatrix[6]*(float)j + 
	// 							homographyMatrix[7]*(float)i + 
	// 							homographyMatrix[8]*1.;
	// 				x = u / w;	/** New x **/
	// 				y = v / w;	/** New y **/
					
	// 				/**
	// 				 * Complete the space remained by corespondances
	// 				 **/
	// 				if (prev_x > x)
	// 				{
	// 					prev_x = x;
	// 				}
	// 				for (int a_x = prev_x; a_x <= x; a_x++)
	// 				{
	// 					if (x > 0 && x < width && y > 0 && y < height)
	// 					{
	// 						orthogonal[y*width + a_x] = perspective_image.data[i*perspective_image.step[0] + j];
	// 					}
	// 				}
	// 				prev_x = x;	
	// 			}
	// 			/**
	// 			 * Complete the space remained by corespondances
	// 			 **/
	// 			for (int a_y = prev_y+1; a_y < y; a_y++)
	// 			{
	// 				for (int a_x = 0; a_x < width; a_x++)
	// 				{
	// 					orthogonal[a_y*width + a_x] = (orthogonal[prev_y*width + a_x] + orthogonal[prev_y*width + a_x])/2;
	// 				}
	// 			}
	// 			prev_y = y;
	// 			j++;
	// 		}
	// 		i++;
	// 	}

	// }
}

/**
 * ################################################################
 * 					PERSPECTIVE WRAPING
 * ################################################################
 * 
 * 	This function translate a selected plane from an image into a destination plane
 * using a Homography matrix. This is done by multipling every point with
 * the Homography matrix to optain the new point
 **/
void perspectiveWrap(Mat perspective_image, uchar *orthogonal, float* homographyMatrix, int width, int height)
{
	int x;
	int y;
	/**
	 * For every pixel in the selected region aply the transformation
	 **/
	// imshow("abcd", perspective_image);
	for (int i = 5; i < height-5; i++)
	{
		for (int j = 5; j < width-5; j++)
		{
			{
				float u = 	homographyMatrix[0]*(float)j + 
							homographyMatrix[1]*(float)i + 
							homographyMatrix[2]*1.;
				float v = 	homographyMatrix[3]*(float)j + 
							homographyMatrix[4]*(float)i + 
							homographyMatrix[5]*1.;
				float w = 	homographyMatrix[6]*(float)j + 
							homographyMatrix[7]*(float)i + 
							homographyMatrix[8]*1.;
				x = u / w;	/** New x **/
				y = v / w;	/** New y **/

				// cout << "u -> x: " << x << " -> " << i << endl;
				// cout << "v -> y: " << y << " -> " << j << endl;
				
				/**
				 * Complete the space remained by corespondances
				 **/
				orthogonal[i*width + j] = perspective_image.at<uchar>(round(y), round(x));
				// printf("Perspe (%d, %d) = %d\n", y, x, perspective_image.at<uchar>((int)(y), (int)(x)));
				// printf("Orthog (%d, %d) = %d\n", j, i, orthogonal[i*width + j]);
			}
			/**
			 * Complete the space remained by corespondances
			 **/
			// for (int a_y = prev_y+1; a_y < y; a_y++)
			// {
			// 	for (int a_x = 0; a_x < width; a_x++)
			// 	{
			// 		orthogonal[a_y*width + a_x] = (orthogonal[prev_y*width + a_x] + orthogonal[prev_y*width + a_x])/2;
			// 	}
			// }
			// prev_y = y;
			// j++;
		}
		// i++;
	}

}

/**
 * ################################################################
 * 					PERSPECTIVE POINTS WRAPING
 * ################################################################
 * 
 * 	This function translate number of points from a selected plane from an image into a destination plane
 * using a Homography matrix. This is done by multipling every point with
 * the Homography matrix to optain the new point
 **/
void perspectivePoints(Point2d* src_points, Point2d* dst_points, float* homographyMatrix, int pointsNo)
{
	for (int i = 0; i < pointsNo; i++)
	{
		float u =	homographyMatrix[0]*(float)src_points[i].x +
					homographyMatrix[1]*(float)src_points[i].y +
					homographyMatrix[2]*1.;

		float v =	homographyMatrix[3]*(float)src_points[i].x +
					homographyMatrix[4]*(float)src_points[i].y +
					homographyMatrix[5]*1.;

		float w =	homographyMatrix[6]*(float)src_points[i].x +
					homographyMatrix[7]*(float)src_points[i].y +
					homographyMatrix[8]*1.;
		dst_points[i].x = int(u/w);
		dst_points[i].y = int(v/w);
	}

}

/**
 * ################################################################
 * 					DISTANCE FROM A POINT TO A LINE
 * ################################################################
 **/
float distance(float a, float b, float c, Point2d p)
{
	return abs(a*p.x + b*p.y + c)/sqrt(a*a + b*b);
}

/**
 * ################################################################
 * 					RANDOM SAMPLE CONSENSUS (RANSAC)
 * ################################################################
 *
 * 	This function finds the line that has the most points near it.
 * 	Two points are randomly selected and then is computed the distance from
 * all the others points to the line resulted from the initial points. Every poin
 * is counted only if the distance is smaller then a given value. The algorithm is
 * repeted until we count enought point in the line closure or until we made a
 * preselected number of iterations
 **/
void RANSAC(Mat src, Mat dst, vector<Point2d> points)
{
	// Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
	int pointsNo = 0;
	float a,b,c; a=b=c=0.;

	Point2d p0;
	Point2d p1;

	/**
	 * While there are less then 250 points in the line closure repeat the algorithm
	 **/
	while (pointsNo < 250 && !points.empty())
	{
		vector<Point2d> points_tmp = points;
		int sum = 0;

		/**
		 * Pick 2 points randomly
		 **/
		int pointIndex0 = rand() % points_tmp.size();
		p0 = points_tmp.at(pointIndex0);
		points_tmp.erase(points_tmp.begin()+pointIndex0, points_tmp.begin()+pointIndex0+1);
		
		int pointIndex1 = rand() % points_tmp.size();
		p1 = points_tmp.at(pointIndex1);
		points_tmp.erase(points_tmp.begin()+pointIndex1, points_tmp.begin()+pointIndex1+1);
		

		/**
		 * Compute the parameter for line determined by the 2 points
		 **/
		float a_tmp = p1.y - p0.y;
		float b_tmp = p1.x - p0.x;
		float c_tmp = p0.x*p1.y - p1.x*p0.y;

		/**
		 * For all points compute the distance from them to the line. If the distance
		 * is smaller than 1.5 then rise the points contor.
		 **/
		while (!points_tmp.empty())
		{
			int pointXIndex = rand() % points_tmp.size();
			Point2d x = points_tmp.at(pointXIndex);
			points_tmp.erase(points_tmp.begin()+pointXIndex, points_tmp.begin()+pointXIndex+1);
			// x = points_tmp.front(); points_tmp.erase(points_tmp.begin(), points_tmp.begin()+1);
			if (distance(a_tmp,b_tmp,c_tmp,x) < 1.5)
			{
				// cout << "pp0: " << x << " dist: " << distance(a_tmp,b_tmp,c_tmp,x) << endl;
				sum++;
			}
		}
		if (sum > pointsNo)
		{
			pointsNo = sum;
		}
	}
	cout << "points: " << pointsNo << endl;
	cout << "p0: " << p0 << " p1: " << p1 << endl;
	// int top, bottom, left, right;
	// top = (int) (0.05*src.rows); bottom = (int) (0.05*src.rows);
	// left = (int) (0.05*src.cols); right = (int) (0.05*src.cols);
	// copyMakeBorder( src, src, top, bottom, left, right, BORDER_CONSTANT);
	cvtColor(src, dst, COLOR_GRAY2BGR);
	line(dst,{p0.y,p0.x}, {p1.y,p1.x},(255,255,255),5);
	
	imshow("ransac", dst);
	waitKey(0);
}

/**
 * ################################################################
 *	 					STREET LINES FILTER
 * ################################################################
 *
 * 	Filter the street lines by converting the image in HSL (Hue, Saturation, Lightness)
 * color model and splitting it in the 3 channels. By observation, the white lines has
 * a big Lightness and a small value in the other 2 chanels. So the formula for fitering
 * the lines is: lightness(px) > hue(px) + saturation(px).
 **/
void lineFilter(Mat src, uchar* dst, int width)
{
	Mat hsl;
	Mat hsl_channels[3];
	cvtColor(src, hsl, COLOR_RGB2HLS);
	split(hsl, hsl_channels);

	// Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

	for (int i = 480; i < 660; i++)
	{
		for (int j = 150; j < 1060; j++)
		{
			if (hsl_channels[1].data[hsl_channels[1].step[0]*i + j] > 
				hsl_channels[0].data[hsl_channels[0].step[0]*i + j] + 
				hsl_channels[2].data[hsl_channels[2].step[0]*i + j])
			{
				// dst[width*i + j] = hsl_channels[1].data[hsl_channels[1].step[0]*i + j];
				dst[width*i + j] = 255;
			}
		
		}
	}

	// imshow("hsl_channels1", hsl_channels[1]);
	// imshow("hsl_channels0", hsl_channels[0]);
	// imshow("hsl_channels2", hsl_channels[2]);
	// imshow("srca", src);
	// waitKey(0);
}

/**
 * ################################################################
 * 					COMPUTE THE DISPLACEMENT 
 * 						FROM THE CENTER 
 * 						  OF THE LANE
 * ################################################################
 **/
float getDisplacement(vector<Point2d> points)
{
	float D = points[0].x - points[1].x;
	float L = abs(points[2].x - -points[3].x);
	float percent = (D*100.)/L;
	return percent;
}

float getAngle(vector<Point2d> points)
{
	double x1 = (float)points[0].x - points[1].x;
	double y1 = (float)points[0].y - points[1].y;
	double x2 = (float)points[2].x - points[3].x;
	double y2 = (float)points[2].y - points[3].y;
	//
	double angle1 , angle2 , angle;
	//
	if (x1 != 0.0f)
		angle1 = atan(y1/x1);
	else
		angle1 = PI/2.; // 90 degrees
	//
	if (x2 != 0.0f)
		angle2 = atan(y2/x2);
	else
		angle2 = PI/2.; // 90 degrees
	//
	angle = fabs(angle2-angle1);
	angle = angle * 180.0 / PI; // convert to degrees ???
	//
	return angle;
}

int main(int argc, char** argv)
{
    Mat src;
	Mat src_clone;
	Mat hsl;
	Mat filteredLine;
	Mat highlight;
	
	Mat filtered_birdeye	= Mat::zeros(480, 640, CV_8UC1);
	Mat edges 				= Mat::zeros(480, 640, CV_8UC1);
	Mat birdeye 			= Mat::zeros(480, 640, CV_8UC1);
	Mat dst3 				= Mat::zeros(480, 640, CV_8UC3);

    char *url = "vid0.mp4";

    cv::VideoCapture cap(url);

	/** Points in the street plane **/
    Point2f src_vertices[4];
	{
		src_vertices[0] = cv::Point(530, 480);
		src_vertices[1] = cv::Point(750, 480);
		src_vertices[2] = cv::Point(1060, 660);
		src_vertices[3] = cv::Point(150, 660);
	}

	/** Points in the birdeye view plane **/
	cv::Point2f dst_vertices[4];
	{
		dst_vertices[0] = cv::Point(0, 0);
		dst_vertices[1] = cv::Point(640, 0);
		dst_vertices[2] = cv::Point(640, 480);
		dst_vertices[3] = cv::Point(0, 480);
	}

	/** Homography matrices **/
	Mat street_to_birdeye = getHomographyMatrix(src_vertices, dst_vertices);
	Mat birdeye_to_street = getHomographyMatrix(dst_vertices, src_vertices);
    while (cap.isOpened())
    {
        cap >> src;
		cap >> src_clone;

		filteredLine	= Mat::zeros(src_clone.rows, src_clone.cols, CV_8UC1);
		Mat alphaShape	= Mat::zeros(src.rows, src.cols, CV_8UC3);

		lineFilter(src_clone, filteredLine.data, filteredLine.cols);
		// perspectiveWrap(filteredLine, birdeye.data, (float*)street_to_birdeye.data, 150, 1060, 490, 770, 640, 480);
		perspectiveWrap(filteredLine, birdeye.data, (float*)birdeye_to_street.data, 640, 480);
		// warpPerspective(filteredLine,birdeye,street_to_birdeye,birdeye.size() );
		// gaussian_noise_Y(birdeye.data, filtered_birdeye.data, 640, 480, 5);
		smoothingFilter(birdeye, filtered_birdeye.data, 640, 480, 5);
		smoothingFilter(filtered_birdeye, birdeye.data, 640, 480, 5);
		smoothingFilter(birdeye, filtered_birdeye.data, 640, 480, 5);
		vector<Point2d> points = edgesDetection(filtered_birdeye.data, edges.data, 640, 480, 0, 480, 50);
		// cout << points << endl;
		
		Point2d src_points[6];
		Point2d dst_points[6];

		copy(points.begin(), points.end(), src_points);
		perspectivePoints(src_points, dst_points, (float*)birdeye_to_street.data, 6);

		// cout << "dst: " << dst_points[0] << ", " << dst_points[1] << ", "<< dst_points[2] << ", "<< dst_points[3] << ", "<< dst_points[4] << ", "<< dst_points[5] << endl;

		vector<Point> polygon(begin(dst_points), end(dst_points)-2);
		iter_swap(polygon.begin()+2, polygon.begin()+3);
		fillConvexPoly(alphaShape, polygon, (0,0,255));
		addWeighted(src, 0.7, alphaShape, 0.3, 0.5, src);
		// fillPolygon(src, dst_points);

		line(src, dst_points[0], dst_points[1], (0,0,255), 4);
		line(src, dst_points[2], dst_points[3], (0,0,255), 4);
		line(src, dst_points[4], dst_points[5], (0,125,255), 4);
		// line(src, Point2d(src.cols/2, 0), Point2d(src.cols/2, src.rows-1), (100,125,100), 4);
		
		vector<Point2d> displacement_points;
		displacement_points.push_back(Point2d(filtered_birdeye.cols/2, 0));
		displacement_points.push_back(src_points[5]);
		displacement_points.push_back(src_points[1]);
		displacement_points.push_back(src_points[3]);

		vector<Point2d> angle_points;
		angle_points.push_back(Point2d(src_points[5].x, filtered_birdeye.rows-1));
		angle_points.push_back(src_points[5]);
		angle_points.push_back(src_points[4]);
		angle_points.push_back(src_points[5]);
		float displacement = getDisplacement(displacement_points);
		// float angle = getAngle(angle_points);
		putText(src, to_string(displacement) + '%', Point(src.cols/2, src.rows-1), 5, 2, Scalar(0,50,255), 2);
		// putText(src, to_string((int)angle), Point(100, src.rows-5), 5, 2, Scalar(0,50,255), 2);


		// RANSAC(dst1, dst3, points);
        imshow("filtered_birdeye", filtered_birdeye);
		imshow("birdeye", birdeye);
        imshow("edges", edges);
		imshow("filteredLine", filteredLine);
		imshow("src", src);
		#ifdef keyboard
        	waitKey(0);
		#else
			waitKeyEx(33);
		#endif
    }
}