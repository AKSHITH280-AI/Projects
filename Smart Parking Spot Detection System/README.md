# Car Parking Space Detection Project

This project demonstrates a simple car parking space detection system using OpenCV and cvzone. It allows you to mark parking spaces on a static image, and then it detects the occupancy of these spaces in a video feed.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Files and Folders](#files-and-folders)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)

## Overview

The main goal of this project is to detect and monitor car parking spaces. It consists of two main parts:

1. **parkingspacepicker.py**: This Python script allows you to manually select parking space coordinates on a static image (`carParkImg.png`). You can left-click to mark parking spaces and right-click to remove them. The coordinates are saved in a file named `CarParkPos` using pickle.

2. **main.py**: This Python script reads the saved parking space coordinates and processes a video feed (`carPark.mp4`) to detect the occupancy of parking spaces. It displays the video with marked parking spaces and updates the count of free spaces in real-time.

## Usage

1. Run `parkingspacepicker.py` to mark the parking spaces on the static image. Left-click to add parking spaces and right-click to remove them. The coordinates will be saved in `CarParkPos`.

2. Run `main.py` to start processing the video feed and detecting parking space occupancy. The video will display with marked parking spaces and a count of free spaces.

## Files and Folders

- `carPark.mp4`: Input video file containing the parking lot footage.
- `carParkImg.png`: Static image of the parking lot for marking parking spaces.
- `parkingspacepicker.py`: Script to select and save parking space coordinates.
- `main.py`: Script for processing the video and detecting parking space occupancy.
- `CarParkPos`: Binary file containing saved parking space coordinates (created by `parkingspacepicker.py`).

