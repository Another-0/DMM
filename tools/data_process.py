import os
import cv2
from tqdm import tqdm


def create_file(output_dir_vi, output_dir_ir):
    if not os.path.exists(output_dir_vi):
        os.makedirs(output_dir_vi)
    if not os.path.exists(output_dir_ir):
        os.makedirs(output_dir_ir)
    print(f"Created folder: ({output_dir_vi}); ({output_dir_ir})")


def update(input_img_path, output_img_path, is_infrared=False):
    image = cv2.imread(input_img_path)
    cropped = image[100:612, 100:740]  # 裁剪坐标为[y0:y1, x0:x1]
    if is_infrared:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_img_path, cropped)


# 主循环
data_root = "data/dv"
output_root = "data/dv"

for cls in ["train", "test", "val"]:
    dataset_dir_vi = output_root + f"/{cls}/{cls}img"
    dataset_dir_ir = output_root + f"/{cls}/{cls}imgr"

    output_dir_vi = data_root + f"/{cls}/image_vi"
    output_dir_ir = data_root + f"/{cls}/image_ir"

    create_file(output_dir_vi, output_dir_ir)

    image_filenames_vi = [(os.path.join(dataset_dir_vi, x), os.path.join(output_dir_vi, x).replace(".jpg", ".png")) for x in os.listdir(dataset_dir_vi)]
    image_filenames_ir = [(os.path.join(dataset_dir_ir, x), os.path.join(output_dir_ir, x).replace(".jpg", ".png")) for x in os.listdir(dataset_dir_ir)]

    # 转化所有可见光图像
    print("Start transforming vision images...")
    for path in tqdm(image_filenames_vi):
        update(path[0], path[1])

    # 转化所有红外图像
    print("Start transforming infrared images...")
    for path in tqdm(image_filenames_ir):
        update(path[0], path[1], is_infrared=True)
