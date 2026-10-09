# Handwritten Digit Recognizer (CNN + FastAPI)

A deep learning web app that recognizes digits drawn on screen. A CNN trained on MNIST (99.19% test accuracy) is served with FastAPI and has a dark, mobile-friendly drawing-pad UI.



![Demo](demo.jpg)



## Features
- Draw a digit with a finger or mouse and get a live prediction
- Probability bars for all 10 digits
- Smart preprocessing: the drawing is cropped and centered in 28x28, like MNIST images
- REST API with auto-generated docs at `/docs`

## Model comparison

| Model | Test accuracy |
|---|---|
| Dense | 97.43% |
| LSTM | 97.75% |
| **CNN** | **99.19%** |

I also analyzed the CNN with a confusion matrix and its wrong predictions. Most errors are messy or ambiguous digits.

## Tech stack
Python, TensorFlow/Keras, FastAPI, Uvicorn, Pillow, NumPy, Matplotlib, Seaborn

## Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --port 8000
```
Open http://localhost:8000

## Project files
- `app.py`: FastAPI backend and UI
- `mnist_cnn.keras`: trained CNN model
- `requirements.txt`: dependencies
