import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm


def qbox2rbox(box):
    if box.shape != (8,):
        raise ValueError("Input box must have 8 elements.")

    points = box.reshape(4, 2)
    (x, y), (w, h), angle = cv2.minAreaRect(points)
    angle_radians = angle / 180 * np.pi
    rbox = np.array([x, y, w, h, angle_radians])
    return rbox


def rbox2qbox(box):
    if box.shape != (5,):
        raise ValueError("Input box must have 5 elements.")

    centerx, centery, w, h, theta = box
    cosa = np.cos(theta)
    sina = np.sin(theta)

    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = h / 2 * sina, h / 2 * cosa

    p1x, p1y = centerx - wx + hx, centery - wy - hy
    p2x, p2y = centerx + wx + hx, centery + wy - hy
    p3x, p3y = centerx + wx - hx, centery + wy + hy
    p4x, p4y = centerx - wx - hx, centery - wy + hy

    return np.array([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y])


# 中心点坐标、矩形的宽和高、旋转角（单位是角度，不是弧度）
def iou_rotate_calculate(box1, box2):
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
    r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        intersec = cv2.contourArea(order_pts)
        union = area1 + area2 - intersec
        iou = intersec * 1.0 / union
    else:
        iou = 0.0
    return iou


def voc_to_dota(xml_dir, xml_name, txt_path):
    txt_name = xml_name[:-4] + ".txt"  # txt文件名字：去掉xml 加上.txt
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    txt_file = os.path.join(txt_path, txt_name)  # txt完整的含名文件路径
    xml_file = os.path.join(xml_dir, xml_name)

    tree = ET.parse(os.path.join(xml_file))  # 解析xml文件 然后转换为DOTA格式文件
    root = tree.getroot()
    with open(txt_file, "w", encoding="UTF-8") as out_file:
        # out_file.write('imagesource:null' + '\n' + 'gsd:null' + '\n')
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name == "feright car" or name == "feright_car" or name == "feright" or name == "*":
                name = "freight-car"
            else:
                name = name

            obj_difficult = obj.find("difficult")
            if obj_difficult:
                difficult = obj_difficult.text
            else:
                difficult = "0"

            if obj.find("bndbox"):
                obj_bnd = obj.find("bndbox")
                obj_xmin = int(obj_bnd.find("xmin").text) - 100
                obj_ymin = int(obj_bnd.find("ymin").text) - 100
                obj_xmax = int(obj_bnd.find("xmax").text) - 100
                obj_ymax = int(obj_bnd.find("ymax").text) - 100

                x1 = obj_xmin
                y1 = obj_ymin
                x2 = obj_xmax
                y2 = obj_ymin
                x3 = obj_xmax
                y3 = obj_ymax
                x4 = obj_xmin
                y4 = obj_ymax

            elif obj.find("polygon"):
                obj_bnd = obj.find("polygon")
                x1 = int(obj_bnd.find("x1").text) - 100
                x2 = int(obj_bnd.find("x2").text) - 100
                x3 = int(obj_bnd.find("x3").text) - 100
                x4 = int(obj_bnd.find("x4").text) - 100
                y1 = int(obj_bnd.find("y1").text) - 100
                y2 = int(obj_bnd.find("y2").text) - 100
                y3 = int(obj_bnd.find("y3").text) - 100
                y4 = int(obj_bnd.find("y4").text) - 100

            data = str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(x3) + " " + str(y3) + " " + str(x4) + " " + str(y4) + " "
            data = data + name + " " + difficult + "\n"
            out_file.write(data)


if __name__ == "__main__":

    data_root_ori = f"data/dv"
    data_root = f"data/dv"

    for cls in ["train", "val", "test"]:
        label_dir_vi = data_root_ori + f"/{cls}/{cls}label"
        output_dir_vi = data_root + f"/{cls}/annofiles_vi"

        label_dir_ir = data_root_ori + f"/{cls}/{cls}labelr"
        output_dir_ir = data_root + f"/{cls}/annofiles_ir"

        xmlFile_list = os.listdir(label_dir_vi)
        for i in tqdm(range(0, len(xmlFile_list))):
            voc_to_dota(label_dir_vi, xmlFile_list[i], output_dir_vi)

        xmlFile_list = os.listdir(label_dir_ir)
        for i in tqdm(range(0, len(xmlFile_list))):
            voc_to_dota(label_dir_ir, xmlFile_list[i], output_dir_ir)
