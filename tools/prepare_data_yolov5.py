import os
import cv2
import csv
import glob
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-dir", type=str)
parser.add_argument("-o", "--output-dir", type=str, default="njord-yolo")
parser.add_argument("-e", "--extract-every-n-frames", type=int, default=25)

CLASS_ID_MAPPING = {
    "boat": 0,
    "person": 1,
    "net": 2,
    "fish": 3,
}

def prepare_data_yolov5(input_dir, output_dir, extract_every_n_frames):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dirpath in glob.glob(os.path.join(input_dir, "videos", "*")):

        video_name = os.path.basename(dirpath)

        if video_name == "unannotated":
            continue

        print("Preparing %s..." % video_name)

        video_filepath = os.path.join(dirpath, "%s.mp4" % video_name)
        video_bb_filepath = os.path.join(dirpath, "%s_bb.csv" % video_name)

        if not os.path.exists(video_bb_filepath):
            continue
        
        video_label_output_path = os.path.join(output_dir, video_name, "labels")
        video_frame_output_path = os.path.join(output_dir, video_name, "images")
        
        if not os.path.exists(video_label_output_path):
            os.makedirs(video_label_output_path)
            
        if not os.path.exists(video_frame_output_path):
            os.makedirs(video_frame_output_path)

        video_labels = defaultdict(list)

        video = cv2.VideoCapture(video_filepath)

        if extract_every_n_frames is None:
            extract_every_n_frames = 0

        extracted_frame_ids = []
        frame_index = 0

        while(video.isOpened()):

            ret, frame = video.read()

            if ret == False:
                break

            if frame_index % extract_every_n_frames == 0:
                extracted_frame_ids.append(frame_index)
                frame_filepath = os.path.join(video_frame_output_path, "%s_frame_%i.jpg" % (video_name, frame_index))
                cv2.imwrite(frame_filepath, frame)

            frame_index += 1

        video.release()

        with open(video_bb_filepath) as f:
            csv_reader = csv.reader(f, delimiter=";")
            _ = next(csv_reader)

            for line in csv_reader:

                frame_id = int(line[0])
                class_name = line[1]

                if frame_id not in extracted_frame_ids:
                    continue

                class_id = CLASS_ID_MAPPING[class_name]

                bb_x = line[2]
                bb_y = line[3]
                bb_width = line[4]
                bb_height = line[5]

                video_labels[frame_id].append([class_id, bb_x, bb_y, bb_width, bb_height])

        framepath_file = open(os.path.join(output_dir, video_name, "%s.txt" % video_name), "w")

        for frame_index, frame_labels in video_labels.items():
            frame_save_path = os.path.join(video_label_output_path, "%s_frame_%i.txt" % (video_name, frame_index))
            framepath_file.write(frame_save_path + "\n")
            with open(frame_save_path, "w") as f:
                for frame_label in frame_labels:
                    f.write(" ".join([str(val) for val in frame_label]) + "\n")
        framepath_file.close()

if __name__ == "__main__":

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    extract_every_n_frames = args.extract_every_n_frames

    prepare_data_yolov5(
        input_dir=input_dir,
        output_dir=output_dir,
        extract_every_n_frames=extract_every_n_frames)


