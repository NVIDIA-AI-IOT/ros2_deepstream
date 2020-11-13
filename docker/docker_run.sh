sudo xhost +si:localuser:root
sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY --device="/dev/video0:/dev/video0" -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v ${pwd}:/workdir ros2_deepstream_base:jp44
