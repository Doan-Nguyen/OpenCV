/*     Brightness & contrast adjustment
+ Math:
    - g(x) = a.f(x) + b
*/

#include <iostream>
#include <opencv2/highgui/highgui.hpp>  // show
#include <opencv2/imgcodecs/imgcodecs.hpp>  // reading & writing

using namespace cv;

int main(){
    Mat org_img;
    org_img = imread('../ImageProcessing/thuyDuong.jpg');
    /*      Transformation images
    - initial all pixel values = 0
    - same size & type as the origin images
    */
   Mat new_img = Mat::zeros(org_img.size(), org_img.type());
   //   a = 1.0, b = 0.3
   float a=1.0;
   float b=0.3;
}
