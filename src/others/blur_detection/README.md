# Kiểm tra ảnh CMT/CCCD bị mờ
```
from blur_classifier import BlurClassifer
blur_classifier = BlurClassifier()
# kiểm tra ảnh CMT mặt trước cũ đã crop, return True nếu bị mờ
blur_classifier.classify_blur_old_front_id_card(img, img_type='RGB')
# kiểm tra ảnh CMT mặt trước mới đã crop, return True nếu bị mờ
blur_classifier.classify_blur_new_front_id_card(img, img_type='RGB')
# kiểm tra ảnh CMT mặt sau cũ đã crop, return True nếu bị mờ
blur_classifier.classify_blur_old_back_id_card(img, img_type='RGB')
# kiểm tra ảnh CMT mặt sau mới đã crop, return True nếu bị mờ
blur_classifier.classify_blur_new_back_id_card(img, img_type='RGB')
```