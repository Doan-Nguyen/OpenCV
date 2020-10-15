/*     Brightness & contrast adjustment
+ Math:
    - g(x) = a.f(x) + b
+ Meansure execution time (ms)
    ```
    double duration;
    duration = static_cast<double>(cv::getTickCount());
    function();
    duration = static_cast<double>(cv::getTickCount()) - duration;
    duration /= getTickFrequency();
    ```
*/

#include <iostream>
#include <opencv2/highgui/highgui.hpp>  // show
#include <opencv2/imgcodecs/imgcodecs.hpp>  // reading & writing

using namespace cv;


Mat constrast_image_loop(Mat &org_image, float alpha, float beta){
    /*      
    + Create new image from original image
        - initial all pixel values = 0
        - same size & type as the origin images
    + Transform each pixel in new_iamge
    */
    Mat new_img = Mat::zeros(org_image.size(), org_image.type());
    for (int y=0; y<org_image.rows; y++){
       for (int x=0; x<org_image.cols; x++){
           for(int c=0; c<org_image.channels(); c++){
                new_img.at<Vec3d>(y, x)[c] = saturate_cast<uchar>(alpha*org_image.at<Vec3d>(y, x)[c] + beta);
            }
        }
    }
    return new_img;
}


Mat constrast_image_pointers(Mat &org_image, float alpha, float beta){
    /*      
    + Create new image from original image
        - initial all pixel values = 0
        - same size & type as the origin images
    + Transform each pixel in new_iamge
    */
    
    int numb_y = org_image.rows;
    int numb_x = org_image.cols*org_image.channels();

    for(int y=0; y<numb_y; y++){
        // get address of row 
        uchar* pixel_y = org_image.ptr<uchar>(y);
        for(int x=0; x<numb_x; x++){
            pixel_y[x] = alpha*pixel_y[x] + beta;
        }
    }

    return org_image;
}

// Mat constrast_image_iterators(Mat &org_image, float alpha, float beta){
//     /*      
//     + Create new image from original image
//         - initial all pixel values = 0
//         - same size & type as the origin images
//     + Transform each pixel in new_iamge
//             pixel_y[x] = alpha*pixel_y[x] + beta;

//     */
    
//     Mat_<Vec3b>::const_iterator it=org_image.begin<Vec3b>();
//     Mat_<Vec3b>::const_iterator itend=org_image.end<Vec3b>();
//     Mat_<uchar>::iterator itout = org_image.begin<uchar>();
//     // for each pixel
//     for( ; it!= itend; it++, itout++){
//         *itout = (*it)*alpha + beta;
//     }
//     return org_image;
// }




int main(){
    //  load image
    Mat org_img, new_img;
    org_img = imread("thuyDuong.jpg");
    Mat zero_img = Mat::zeros(org_img.size(), org_img.type());


    //           Transfor
    double duration;
    duration = static_cast<double>(cv::getTickCount()); 
    // new_img = constrast_image_loop(org_img, 0.9, 0.2);   // 0.211689
    new_img = constrast_image_pointers(org_img, 0.5, 0.2); //  0.0298756
    duration = static_cast<double>(cv::getTickCount()) - duration;
    duration /= getTickFrequency();
    //    imshow("Original image", org_img);
    std::cout << "Timing execution: " << duration << std::endl;
    imshow("Transformation image", new_img);

    waitKey(1000);

    return 0;
}
