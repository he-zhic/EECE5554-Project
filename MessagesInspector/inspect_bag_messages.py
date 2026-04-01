import sys
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message

from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from wheel_odometry.msg import Distances
from wheel_control.msg import WheelControl

TOPIC_TYPE_MAP = {
    "/distance": Distances,
    "/nav_cmd": WheelControl,
    "/odom": Odometry,
    "/realsense/imu": Imu,
    "/reach_1/imu": Imu,
    "/reach_2/imu": Imu,
    "/reach_3/imu": Imu,
    "/reach_1/fix": NavSatFix,
    "/reach_2/fix": NavSatFix,
    "/reach_3/fix": NavSatFix,
    "/reach_1/ppk/fix": NavSatFix,
    "/reach_2/ppk/fix": NavSatFix,
    "/reach_3/ppk/fix": NavSatFix,
    "/reach_1/vel": TwistStamped,
    "/reach_2/vel": TwistStamped,
    "/reach_3/vel": TwistStamped,
    "/reach_1/ppk/vel": TwistStamped,
    "/reach_2/ppk/vel": TwistStamped,
    "/reach_3/ppk/vel": TwistStamped,
}

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 inspect_bag_messages.py <bag_dir> <topic_name> [n=5]")
        sys.exit(1)

    bag_dir = sys.argv[1]
    target_topic = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    if target_topic not in TOPIC_TYPE_MAP:
        print(f"Topic {target_topic} not in TOPIC_TYPE_MAP. Add it first.")
        sys.exit(1)

    msg_type = TOPIC_TYPE_MAP[target_topic]

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    names = [x.name for x in topic_types]
    if target_topic not in names:
        print(f"Topic not found in bag: {target_topic}")
        print("Available topics:")
        for x in topic_types:
            print(f"  {x.name} : {x.type}")
        sys.exit(1)

    count = 0
    while reader.has_next() and count < n:
        topic, data, t = reader.read_next()
        if topic != target_topic:
            continue

        msg = deserialize_message(data, msg_type)
        print("=" * 80)
        print(f"[{count+1}] topic={topic}")
        print(f"timestamp_ns={t}")
        print(msg)
        count += 1

    print("=" * 80)
    print(f"Printed {count} messages from {target_topic}")

if __name__ == "__main__":
    main()