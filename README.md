# Simple Kalman Filter

This is an implementation of a Simple Kalman Filter. 

It is meant to be used with a twist msg as a velocity input and an imu msg which was recorded with rosbag on a robot running in a straight line.

## Requirements

Clone this project with:
```
$ git clone https://github.com/braineniac/simple_kalman.git
```

- ROS installation
- matplotlib
- numpy
- pandas

You can install these with: 
```
$ pip install -r requirements.txt
```

## Installation
In your workspace run:
```
$ catkin_make simple_kalman
```

## Use

Check out the --help option with all scripts in the bin folder for details.

## TODO
- Simple Kalman Filter running as a node
- move upscaling from bag to core
- extend to multiple types of message inputs
- use multiple dimensions

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details
