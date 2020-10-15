+ Get image size:
    ```
    y, x, c = org_img.rows, org_img.cols, org_img.channels()
    ```

    - Get image channels:
        ```
        org_img.channels()
        ```
+ Truy cập vào từng điểm ảnh trong opencv c++ có 3 phương thức:
    - Quẻt điểm ảnh sử dụng vòng lặp:

    - Quét ảnh sử dụng con trỏ:

    - Quét điểm ảnh sử dụng 
    ```
    // RGB
    org_img.at<Vec3b>(y, x)[c]
    // gray image (1 channel)
    org_img.at<uchar>(y, x)
    ```
+ Ảnh màu 3 kênh, mỗi điểm ảnh có kiểu dữ liệu **8-bit unsigned char** -> tổng số lượng màu 256^3.


## Kiến thức
+ Biến đổi *gamma*, thực hiện phép biến đổi phi tuyến tính hàm số mũ giữa ảnh đầu vào & ảnh đầu ra giúp thay đổi **độ sáng - brightness**:
    - gamma càng giảm độ sáng càng tăng.

+ Video background subtraction
