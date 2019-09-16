# Kalman Estimator

This project estimates a robot's global x and y states and the robot relative v,a,psi,yaw,dyaw, ddyaw states using an IMU and a joystick signal with the help of a Kalman Filter.


## Requirements

Clone this project with:
```
$ git clone https://github.com/braineniac/kalman_estimator.git
```

### ROS
kinetic or melodic

### Python dependencies
- matplotlib
- numpy
- pandas
- scipy

I would recommend creating a virtualenv, then you can install these with:
```
$ pip install -r requirements.txt
```

## Compiling
In your workspace run:
```
$ catkin_make kalman_estimator
```

## Development
This package is currently used as local data analysis package from rosbags and thesis generation.

It has a standard Kalman Filter and an Extended Kalman Filter with an adaptive process covariance with a window function.


## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details
