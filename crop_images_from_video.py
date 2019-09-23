# -*- coding: utf-8 -*-

import cv2
import os


def get_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(7))


def main():
    video_path = "/home/jw/ML/C2019/data/car.mp4"
    # vdo_name = video_path.split("/")[-1].split(".")[0]
    out_dir = "/home/jw/ML/C2019/data/car/"

    cap = cv2.VideoCapture(video_path)
    all_num_frames = int(cap.get(7))
    if all_num_frames <= 0:
        raise Exception("读取视频失败,请检查视频! 视频地址: ", video_path)

    for idx in range(0, all_num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            img_name = str(idx).zfill(4) + ".jpg"
            out_path = os.path.join(out_dir, img_name)
            cv2.imwrite(out_path, frame)


if __name__ == '__main__':
    main()

