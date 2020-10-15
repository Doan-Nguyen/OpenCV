#include <vector>
#include "color_detector.h"

using namespace cv;
using namespace std;


Mat ColorDetector::process(const Mat& image){
    // re-allocate binary map
    result.create(image.size(), CV_8U);
    // converting to lab color space
    if(useLab){
        cvtColor(image, converted, COLOR_BGR2Lab);
    }
    // get the iterators
    Mat_<Vec3b>::const_iterator it= image.begin<Vec3b>();
    Mat_<Vec3b>::const_iterator itend= image.end<Vec3b>();
    Mat_<uchar>::iterator itout= result.begin<uchar>();
    // for each pixel
    for ( ; it!=itend; ++it, ++itout){
        if(getDistanceToTargetColor(*it) < maxDist){
            *itout=255;
        }
        else{
            *itout=0;
        }
        
    }
    return result;
}