# Processing Color Images with Classes 

## 0. Overview
+ Contents:
    - Comparing colors using the strategy design pattern.
    - Segmenting an image with the GrabCut algorithm.
    - Converting color representations 
    - Representing colors with hue, saturation & brightness.


## 1. So sánh ảnh màu sử dụng thiết kế mẫu.
### 1.1 The stategy design pattern ~ mẫu chiến lược.
+ Thuộc 1 trong 3 nhóm chính của *Design Patterns* là **Behavior pattern* hướng tới thiết kế hành vi đối tượng.
+ Mẫu thiết kế giúp trừu tượng hóa các hành vi (behavior, method, function) của đối tượng thông qua các lớp.
+ Ý tưởng chính:
    - Thay vì thiết kế **kế thừa** một hành vi mới trong class -> tất cả các đối tượng mới sẽ sử dụng/có hành vi này => không cần thiết/tốn sức trong quá trình bảo trì.
    - Giải pháp: đóng gói các phần thay đổi tách riêng độc lập các phần không thay đổi.
        - Tạo bộ lớp nơi chứa các hành vi mới cần cập nhật.
        - Một lớp implement & một lớp interface các hành vi cụ thể => các lớp không biết bất kì implement chi tiết nào của nó.


### 1.2 Cách thực hiện 
+ Cách deployed & used:
```
int main(){
    // 1. Create image processor object
    ColorDetector cdetector;
    // 2. Read input image
    Mat org_img = cv2.imread(img_path);
    // 3. Set input parameters
    cdetector.setTargetColor(1, 2, 3);
    // 4. Process the image & display the results
    nameWindow("results");
    Mat result = cdetect.process(org_img);
    imshow("The result", result);

    waitKey(1000);
    return 0;

}
```


### 1.3 Nó làm việc như thế nào ?
+ Thuật toán đọc qua tất cả các điểm ảnh trên ảnh gốc (*input image*) & thực hiện phép so sánh với ảnh đầu ra (*binary output image*).
+ Thuật toán:
    - $1^st$ step: Settin up the required iterators
        - scanning input image
        - Tính toán khoảng cách d{$pixel_current$, $pixel_target$} bằng *getDistanceToTargetColor()*.
        - So sánh với khoảng cách giới hạn *maxDist* (max distance).
            - d < maxDist : $pixel_output$ = 1 



### 1.4 Mở rộng 
+ Contructor:
    - Trùng tên với class & không trả về kiểu dữ liệu, kể cả void.
    - Có thể được tham số hóa để gán giá trị khởi tạo cho đối tượng mới.
        ```
        ColorDetector();
        ```
+ Decontructor:
    - Có nhiệm vụ giải phóng resource khi kết thúc chương trình;
    - Trùng tên với class nhưng có kí hiệu *~* đứng trước:
        ```
        ~ColorDetector();
        ```
+ Kí hiệu
    - vd: int &a; 
        - // khi chương trình chạy trong hàm, nếu biến có thay đổi thì ra ngoài biến sẽ mang giá trị mới & ngược lại nếu không có, sau khi chạy xong hàm ra ngoài biến sẽ mang giá trị mặc định.
        - **int &a; <~> int& a;**
    
    - Const member functions in c++
        - Khai báo giúp hàm không thay đổi trong thiết kế *stategy design pattern*.
        ```
        datatype function_name const();
        ```
        - Thường dùng cho các hàm có vai trò tính toán giá trị.

    - *absdiff()*: computes the absolute difference between the pixels of an image & other object (in this case: a scalar value)

    - *spit(1x3-channel_image, 3x1-channel_images)*: 
        - Thực hiện xử lý ảnh trên các kênh ảnh khác nhau, tránh sự ảnh hưởng giữa các kênh ảnh.

    - threshold(): 
        - So sánh các điểm pixels với giá trị threshold.
        - Các model của threshold:
            - THRESH_BINARY: gán giá trị max value (255) có các điểm ảnh > ngưỡng giá trị (threshold), các giá trị < sẽ gán = 0
            - THRESH_BINARY_INV: ngược lại THRESH_BINARY

    - *floodFill()*:
        - Xác định sự tương đồng về màu sắc của điểm ảnh trên ảnh gốc & target. Ở đây, thuật toán sẽ tìm các điểm ảnh lân cận điểm ảnh được chỉ định & tô màu cho các điểm ảnh lân cận.

    - *[useLab](http://www.astrosurf.com/jwisn/improving.htm#:~:text=%22Lab%22%20color%20model%20lets%20you,into%20a%20modified%20color%20image.)*: 
        - chế độ màu cho phép ảnh được tách thành các kênh riêng biệt & có thể gộp 3 kênh trở lại.