# face-mask-detector
Real-time face mask detection with facial recognition. The program detects whether if someone is wearing a mask or not with the video stream. If a person is found to be not wearing any face mask, a snapshot of that person will be taken, and facial recognition will be done on the snapshot to identify the identity of the person. The information will then be be added into a spreadsheert report. The authorities may then use the information collected in the spreadsheet to send out warnings to people who are caught not wearing any masks.

# Session Information (Training the model)

matplotlib          3.5.2<br />
numpy               1.19.5<br />
session_info        1.0.0<br />
sklearn             1.1.1<br />
tensorflow          2.5.0<br />

-----

IPython             8.4.0<br />
jupyter_client      7.3.1<br />
jupyter_core        4.10.0<br />
notebook            6.4.11<br />

-----

Python 3.9.13 | packaged by conda-forge | (main, May 27 2022, 17:00:33) [Clang 13.0.1 ]
macOS-12.0.1-arm64-arm-64bit

-----

### Session information updated at 2022-06-03 01:00

# Session Information (Running the detector)

cv2                 4.5.5<br />
imutils             0.5.4<br />
keras               2.8.0<br />
numpy               1.22.4<br />
session_info        1.0.0<br />

-----

Python 3.10.4 | packaged by conda-forge | (main, Mar 24 2022, 17:42:03) [Clang 12.0.1 ]
macOS-12.0.1-arm64-arm-64bit

-----

### Session information updated at 2022-06-03 01:14

# Datasets
Credits to Maskedface-Net for the no mask and masked dataset to train the model: https://github.com/cabani/MaskedFace-Net

# How-To
1. Run the "face mask detector trainer/trainer.py" to train the model with the loaded dataset. <br />
2. Run the detect_mask_video.py to start the face mask recognition through your webcam. <br />
   If you are not wearing any mask for more than 3 seconds, the program will take a snapshot. <br />
3. Run the detect_mask_photo to detect any face mask worn in a picture. <br />
3. Input any face data with the person's name as the image's name in the face_ID folder. <br />
4. Run the identifier.py to detect the person not wearing a mask in the snapshot folder. <br />
5. Check the report.csv . <br />
