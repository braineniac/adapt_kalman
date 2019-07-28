# Adaptive Kalman Filter

This is an implementation of a Kalman Filter with adaptive covariance properties.

It is meant to be used with a twist msg as a velocity input and an imu msg which was recorded with rosbag on a robot running in a straight line.

## Requirements

Clone this project with:
```
$ git clone https://github.com/braineniac/adapt_kalman.git
```

- ROS installation
- matplotlib
- numpy
- pandas

You can install these with:
```
$ pip install -r requirements.txt
```

## Compiling
In your workspace run:
```
$ catkin_make adapt_kalman
```

## Use

Check out the --help option with all scripts in the bin folder for details.

## TODO
- extend to multiple types of message inputs
- use multiple dimensions

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details
