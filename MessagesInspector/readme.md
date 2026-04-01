use inspect_bag_messages.py to inspect rosbag(ROS2) messages
command:

python3 inspect_bag_messages.py \
TargetFolder \
/TopicName MessageCount


example:

python3 inspect_bag_messages.py \
"/mnt/hgfs/eece5554/Project/rosariov2数据/2023-12-26-15-10-15_ros2" \
/distance 10
