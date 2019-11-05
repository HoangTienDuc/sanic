# Errors

<aside class="notice">
Phần này bao gồm các lỗi thường gặp và có thể xử lý bởi hệ thống, messages được trả về rõ ràng và cụ thể mục đích nhằm chỉ thị cho users sử dụng API một cách chính xác nhất.
</aside>

Hệ thống FVI sử dụng các error code như sau:


Error Code | Meaning
---------- | -------
1 | **Invalid Parameters or Values!** -- Sai thông số trong request (ví dụ không có key hoặc ảnh trong request body).
2 | **Failed in cropping** -- CMT trong ảnh bị thiếu góc nên không thể crop về dạng chuẩn.
3 | **Unable to find ID card in the image** -- Hệ thống không tìm thấy CMT trong ảnh hoặc ảnh có chất lượng kém (quá mờ, quá tối/sáng).
5 | **No URL in the request** -- Request sử dụng key image_url nhưng giá trị bỏ trống.
6 | **Failed to open the URL!** -- Request sử dụng key image_url nhưng hệ thống không thể mở được URL này.
7 | **Invalid image file** -- File gửi lên không phải là file ảnh.
8 | **Bad data** -- File ảnh gửi lên bị hỏng hoặc format không được hỗ trợ.
9 | **No string base64 in the request** -- Request sử dụng key image_base64 nhưng giá trị bỏ trống.
10 | **String base64 is not valid** -- Request sử dụng key image_base64 nhưng string cung cấp không hợp lệ.
