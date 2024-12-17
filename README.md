# **Fall Detection System Using YOLOv8 and ST-GCN**

## **Project Overview**
This project implements a fall detection system that combines **YOLOv8** for object detection and **ST-GCN (Spatio-Temporal Graph Convolutional Network)** for analyzing human pose keypoints. The system detects falls in videos and sends notifications to caregivers using Twilio while uploading fall-detected frames to Google Drive.

---

## **Technologies Used**

1. **YOLOv8**: Object detection to identify persons and crop frames containing them.
2. **MediaPipe**: Pose estimation for extracting 33 keypoints per frame.
3. **ST-GCN**: Spatio-Temporal Graph Convolutional Network for classifying frames as:
   - **Fallen**
   - **Not Fallen**
4. **Twilio API**: Sends WhatsApp notifications with fall alerts and Google Drive links.
5. **Google Drive API**: Uploads fall-detected frames for easy access.
6. **Python**: Main programming language for development.
7. **Google Colab**: Used for model training, testing, and deployment.

---

## **Pipeline**

The general flow of the project is as follows:

1. **Input Video**: User uploads a video for analysis.
2. **YOLOv8 Phase**:
   - Detects persons and extracts bounding boxes from video frames.
   - Crops and stores the bounding box regions in the **`cropped_frames`** folder.
3. **Pose Extraction**:
   - Extracts 33 keypoints from the cropped frames using **MediaPipe Pose**.
4. **ST-GCN Model**:
   - Trained on keypoints and labels (`fallen`/`not fallen`).
   - Classifies frames using extracted keypoints.
   - Generates:
     - **Per-frame Results**: Labels for each frame.
     - **Overall Result**: Video classification based on frame results.
5. **Twilio Integration**:
   - Sends alerts if a fall is detected.
   - Uploads the detected fall frame to Google Drive and provides a shareable link.

---

## **Logical Flow Diagram**
This figure illustrates the high-level logical flow of the fall detection system:

![Logical Flow Diagram](path/to/logical_flow_diagram.png)  
*Figure 1: Logical flow showing YOLOv8 detection, ST-GCN classification, and notification.*

---

## **Detailed Flow Diagram**
This figure shows the complete system pipeline in detail:

![Detailed Flow Diagram](path/to/detailed_flow_diagram.png)  
*Figure 2: Detailed flow diagram depicting key components like YOLOv8, MediaPipe, ST-GCN, and Twilio integration.*

---

## **Figures and Outputs**
1. **Logical Flow Diagram**: Highlights the overall structure of the project.
2. **Detailed Flow Diagram**: Describes the data flow through all components.
3. **Output Results**:
   - Example per-frame classification:  
     ```
     Frame 1: Not Fallen
     Frame 50: Fallen
     Frame 100: Fallen
     ```
   - Overall Result: **Fall Detected**

---

## **Links to Resources**
- **Dataset**: [Link to the Dataset](#)  
- **Colab Notebook**: [Run on Google Colab](#)  
- **Technical Specification Document**: [Link to Technical Docs](#)  
- **Logical and Flow Diagrams**: [Download Diagrams](#)

---

## **Setup Instructions**

1. Clone the repository:
   ```bash
   git clone <repo-link>
   cd fall-detection-system
   ```

2. Install required libraries:
   ```bash
   pip install torch torchvision ultralytics mediapipe opencv-python twilio google-api-python-client
   ```

3. Run the code in Colab:
   - Upload video files.
   - Use the **`predict_fall_in_video`** function to analyze videos.

4. Example Command:
   ```python
   results = predict_fall_in_video('/content/test_video.mp4', model, device, pose_extractor)
   print(results['Overall Result'])
   ```

---

## **Contributors**
- **[Your Name]**: Model Development, Integration  
- **[Team Member Name]**: Data Processing, Pipeline Development  
- **[Team Member Name]**: Twilio Integration, Cloud Setup  

---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
