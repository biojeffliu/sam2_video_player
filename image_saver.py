import os
import cv2
import numpy as np
import rosbag2_py
import yaml
from tqdm import tqdm
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge

def extract_images(rosbag_path, image_topic, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    bridge = CvBridge()
    
    # Setup ROS 2 bag reader
    storage_options = rosbag2_py.StorageOptions(uri=rosbag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    
    print(type_map)
    msg_type = get_message(type_map[image_topic])
    count = 0

    print(f"Reading from topic: {image_topic}")

    while reader.has_next():
        # if count == 500:
        #     break
        topic, data, t = reader.read_next()
        
        if topic == image_topic:
            msg = deserialize_message(data, msg_type)
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            filename = os.path.join(output_folder, f"{count:05d}.jpg")
            cv2.imwrite(filename, cv_image)
            count += 1

    print(f"Saved {count} images to '{output_folder}'")

if __name__ == "__main__":
    ROSBAG_PATH = "/media/roar-perception/ART-D3/rosbags/lvrc/april_3/run1/rosbag2_2025_04_03-13_54_07"
    IMAGE_TOPIC = "/vimba_rear/image/ptr"  # e.g., "/camera/image_raw"
    OUTPUT_FOLDER = "./apr3_run1_vimba_rear"

    extract_images(ROSBAG_PATH, IMAGE_TOPIC, OUTPUT_FOLDER)
