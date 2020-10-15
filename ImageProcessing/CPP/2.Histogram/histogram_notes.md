# Histogram 

## Histogram dùng để làm gì ?
+ Histogram là biểu đồ thể hiện sự phân bố tập giá trị pixels trong ảnh gốc.    
    - Điều chỉnh sự thay đổi vủa giá trị grey của ảnh.
+ **Histogram equalization**: cân bằng sáng.
    - Giúp điều chỉnh histogram về trạng thái cân bằng.
    - Là phương pháp kéo dãn mật độ phân bố giá trị các điểm ảnh.

## Tác dụng
+ Các tác dụng của việc sử dụng *histogram*:
    - Giảm sự ảnh hưởng do việc quá sáng/tối trong ảnh.
    - *Cân bằng sáng* trong ảnh.

## Histogram tính như thế nào ?
+ Cách tính:
    - $r_k$ mức xám của ảnh f(x, y)
    - $n_k$ số điểm ảnh có giá trị $r_k$
    - Biểu đồ mức xám chưa chuẩn hóa: $h(r_k)=n_k$
    - Biểu đồ chuẩn hóa (normalized histogram): $p(r_k)=\frac{h(r_k)}{W*H}$

+ Giải thuật cân bằng sáng:
    - Tính histogram $H(i)$
    - Chuẩn hóa histogram $H'(i)=\sum^i_(j=0)H(j)$
    - Tính hàm mật độ xác suất:
    - Tính giá trị mức xám từng điểm ảnh: 


## Code 


## Tài liệu tham khảo 
+ Image Processing: The Fundamentals - Maria Petrou and Costas Petrou
+ [Histogram equalization](https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html#how-does-it-work)
+ [Image Processing course - TS.Nguyễn Thị Ngọc Diệp](https://github.com/chupibk/INT3404_1/blob/master/week3/Week%203%20-%20Histogram.pdf)