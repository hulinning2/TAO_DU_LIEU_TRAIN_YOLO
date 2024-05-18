import cv2
import numpy as np
import os

# Hàm để xử lý mỗi tệp ảnh và tạo file labels tương ứng
def process_image(image_path, output_folder):
    # Đọc ảnh
    image = cv2.imread(image_path)

    # Kiểm tra xem ảnh có được đọc thành công hay không
    if image is None:
        print(f"Error: Cannot read image from path '{image_path}'")
        return

    # Chuyển đổi ảnh sang không gian màu HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Định nghĩa phạm vi màu đỏ trong không gian màu HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Định nghĩa phạm vi màu xanh trong không gian màu HSV
    lower_blue1 = np.array([90, 100, 100])
    upper_blue1 = np.array([120, 255, 255])
    lower_blue2 = np.array([120, 100, 100])
    upper_blue2 = np.array([150, 255, 255])

    # Tạo mặt nạ (mask) cho màu đỏ
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Tạo mặt nạ (mask) cho màu xanh
    mask_blue1 = cv2.inRange(hsv_image, lower_blue1, upper_blue1)
    mask_blue2 = cv2.inRange(hsv_image, lower_blue2, upper_blue2)
    mask_blue = cv2.bitwise_or(mask_blue1, mask_blue2)

    # Tìm các đường viền (contours) cho màu đỏ
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Tìm các đường viền (contours) cho màu xanh
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Nếu không có khối màu đỏ hoặc màu xanh, thì thoát khỏi hàm
    if not contours_red and not contours_blue:
        print(f"No red or blue object detected in image '{image_path}'")
        return

    # Nếu chỉ có khối màu đỏ, tạo bounding box và ghi thông tin vào tệp
    if contours_red and not contours_blue:
        largest_contour_red = max(contours_red, key=cv2.contourArea)
        write_bounding_box_info(output_folder, image_path, largest_contour_red, None)

    # Nếu chỉ có khối màu xanh, tạo bounding box và ghi thông tin vào tệp
    elif contours_blue and not contours_red:
        largest_contour_blue = max(contours_blue, key=cv2.contourArea)
        write_bounding_box_info(output_folder, image_path, None, largest_contour_blue)

    # Nếu cả hai màu đỏ và xanh đều có, tạo bounding box và ghi thông tin vào tệp
    else:
        largest_contour_red = max(contours_red, key=cv2.contourArea)
        largest_contour_blue = max(contours_blue, key=cv2.contourArea)
        write_bounding_box_info(output_folder, image_path, largest_contour_red, largest_contour_blue)

# Hàm để ghi thông tin bounding box vào tệp
def write_bounding_box_info(output_folder, image_path, contour_red, contour_blue):
    # Đọc kích thước của ảnh
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Lấy tên của tệp ảnh (bỏ đuôi phần mở rộng)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Tạo đường dẫn cho tệp txt kết quả
    output_path = os.path.join(output_folder, image_name + '.txt')

    # Ghi dữ liệu bounding box vào tệp
    with open(output_path, 'w') as file:
        check = False
        if contour_red is not None and len(contour_red) > 0:
            check = True
            x_red, y_red, w_red, h_red = cv2.boundingRect(contour_red)
            x_center_red = (x_red + w_red / 2) / image_width
            y_center_red = (y_red + h_red / 2) / image_height
            width_red = w_red / image_width
            height_red = h_red / image_height
            file.write(f"0 {x_center_red:.6f} {y_center_red:.6f} {width_red:.6f} {height_red:.6f}")
        if contour_blue is not None and len(contour_blue) > 0:
            x_blue, y_blue, w_blue, h_blue = cv2.boundingRect(contour_blue)
            x_center_blue = (x_blue + w_blue / 2) / image_width
            y_center_blue = (y_blue + h_blue / 2) / image_height
            width_blue = w_blue / image_width
            height_blue = h_blue / image_height
            if check:
                file.write("\n")
            file.write(f"1 {x_center_blue:.6f} {y_center_blue:.6f} {width_blue:.6f} {height_blue:.6f}")

# Thư mục chứa tệp ảnh
image_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IMAGES")

parent_folder = os.path.dirname(image_folder)

# Tạo thư mục "labels" cùng cấp với thư mục "IMAGE"
labels_folder = os.path.join(parent_folder, 'labels')
os.makedirs(labels_folder, exist_ok=True)

# Lặp qua tất cả các tệp ảnh trong thư mục
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        process_image(image_path, labels_folder)

print("Xong")
