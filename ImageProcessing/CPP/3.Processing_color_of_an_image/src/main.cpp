#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "color_detector.h"
#include "color_detector.cpp"

using namespace std;
using namespace cv;


int main(){
    // 1. Creating image processor object
    ColorDetector cdetect;
    // 2. read input image
    Mat org_image = imread("./thuyDuongvtv4.jpg");
    if (org_image.empty()){
        return 0;
    }
    namedWindow("Original image");
    imshow("Original image", org_image);
    waitKey(1000);
    // 3. Set input parameters
    cdetect.setTargetColor(120, 130, 140);
    // 4. Process input image & display the result
    namedWindow("Result");
    Mat result = cdetect.process(org_image);
    imshow("The result image", result);
    waitKey(1000);

    return 0;
}