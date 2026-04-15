# Facial Emotion Recognition

A simple and interactive **Facial Emotion Recognition** system that detects human emotions from images and live video using Deep Learning. This project leverages **DeepFace**, **OpenCV**, and **Streamlit** to provide real-time emotion detection through image uploads, webcam captures, and live camera feeds.

## Features

* Detect emotions from uploaded images
* Capture and analyze emotions using a webcam
* Real-time emotion detection via live camera
* Powered by DeepFace for accurate predictions
* Interactive and user-friendly Streamlit interface
* Displays dominant emotion with confidence scores

## Supported Emotions

The system can recognize the following emotions:

* 😊 Happy
* 😢 Sad
* 😠 Angry
* 😐 Neutral
* 😲 Surprise
* 😨 Fear
* 🤢 Disgust

## Tech Stack

* **Python**
* **DeepFace**
* **OpenCV**
* **Streamlit**
* **NumPy**

## Project Structure

```
FacialEmotionRecognition/
│── app.py                 # Streamlit web application
│── live_camera.py         # Real-time emotion detection using webcam
│── requirements.txt       # Project dependencies
│── README.md              # Project documentation
```

## Installation

Follow these steps to set up the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/harshkad/FacialEmotionRecognition.git
cd FacialEmotionRecognition
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit Web App

```bash
streamlit run app.py
```

This will open the application in your browser.

### Run the Live Camera Emotion Detector

```bash
python live_camera.py
```

Press **Q** to exit the live camera window.

## Future Enhancements

* Deploy on Streamlit Cloud or Hugging Face Spaces
* Add emotion analytics and visualizations
* Support multiple deep learning models
* Emotion history tracking and reporting
* Mobile-friendly interface
