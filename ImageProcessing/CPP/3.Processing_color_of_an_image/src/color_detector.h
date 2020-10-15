#ifndef COLORDETECTOR_H
#define COLORDETECTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


class ColorDetector{
private:
    int maxDist;
    //  target color
    Vec3b target;
    // image containning color converted image
    Mat converted;
    bool useLab;

    // image containning resulting binary map
    Mat result;

public:
    //               Contructor & decontructor 
    // empty constructor: default parameter intialization 
    ColorDetector() : maxDist(100), target(120, 130, 140), useLab(false){}
    // extract constructor for lab color space
    ColorDetector(bool useLab): maxDist(100), target(120, 130, 140), useLab(useLab){}
    // full constructor
    ColorDetector(uchar blue, uchar green, uchar red, int max_dist=100, bool useLab=false): maxDist(max_dist), useLab(useLab){
        setTargetColor(blue, green, red);
    }

    // ~ColorDetector();

    //              Computer the distance between org_image with target color.
    int getDistanceColor(const Vec3b& color1, const Vec3b& color2) const {
        /*  This function computer the distance between 2 colors (3 channels)
        distance = abs(color1[1] - color2[1]) + ... + abs(color1[3] - color2[3])
        */
        int sum = 0;
        for(int i=0; i<3; i++){
            sum += abs(color1[i] - color2[i]);
        }
        return sum;
    }

    int getDistanceToTargetColor(const Vec3b& color) const {
        return getDistanceColor(color, target);
    }

    
    Mat process(const Mat& org_image);

    Mat operator()(const Mat& org_image){
        Mat input_image;
        if (useLab){
            cvtColor(org_image, input_image, COLOR_BGR2Lab);
        }
        else{
            input_image=org_image;
        }
        
        Mat output;
        // compute absolute difference with target color 
        absdiff(input_image, Scalar(target), output);
        // spit the channels into 3 images
        vector<Mat> images;
        split(output, images);
        // add the 3 channels 
        output = images[0] + images[1] + images[2];

        cv::threshold(output,  // input image
                      output,  // output image
                      maxDist, // threshold (must be < 256)
                      255,     // max value
                 cv::THRESH_BINARY_INV); // thresholding type
	
	    return output;
    }

    //              Getters & setters

    // Sets the color distance threshold
    void setColorDistanceThreshold(int distance){
        if(distance<0){
            distance=0;
        }
        maxDist=distance;
    }
    // gets the color distance threshold
    int getColorDistanceThreshold() const{
        return maxDist;
    }
    // Sets the color to be detected given in RGB images
    void setTargetColor(uchar blue, uchar green, uchar red){
        // BGR order 
        target = Vec3b(blue, green, red);

        if(useLab){
            // temporary 1-pixel image
            Mat tmp(1, 1, CV_8UC3);
            tmp.at<Vec3b>(0, 0) = Vec3b(blue, green, red);
            // converting the target to lab color space
            cvtColor(tmp, tmp, COLOR_BGR2Lab);
            target = tmp.at<Vec3b>(0, 0);
        }
    }

    void setTargetColor(Vec3b color){
        target=color;
    }

    Vec3b getTargetColor() const{
        return target;
    }
};

#endif