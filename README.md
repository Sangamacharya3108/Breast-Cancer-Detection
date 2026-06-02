# Breast Cancer Detection from Histopathology Images

This is a web-based project that helps detect breast cancer from microscopic tissue images (also called histopathology slides). Users can upload a tissue image patch, and the system will predict whether the tissue is Benign (non-cancerous) or Malignant (cancerous).

To make the application realistic, we also added an image validation layer. This means if you upload a random picture like a dog, a car, or a selfie, the system will detect that it is not a medical tissue slide and reject it with an error message, instead of giving a wrong cancer prediction.

---

## How It Works

### For Non-Technical Readers
When doctors check for breast cancer, they take a small tissue sample, stain it with dyes (which make it look pink and purple), and look at it under a microscope. This is called a histopathology slide.
This project uses a computer program (a Deep Learning model) that has learned to recognize the difference between healthy tissue patterns and cancer tissue patterns from thousands of past images.
When you upload an image:
1. The system checks if the image has the correct pink/purple color stain and the detailed texture of a microscope slide.
2. If it is a valid slide, it runs it through the trained AI model.
3. It displays the result (Benign or Malignant) along with a confidence percentage (how sure the model is).

### For Technical Readers
1. **Frontend**: Built using HTML5, Bootstrap 5 for responsive design, and Vanilla JavaScript to handle file uploads, drag-and-drop, and the loading animations.
2. **Backend**: Powered by Flask (Python). It handles the routing, session history, and triggers the image processing.
3. **Validation Layer (`histopath_validate.py`)**: Uses a hybrid validation check. If a custom validation model is provided, it runs a MobileNetV2 classifier. Otherwise, it uses a fallback algorithm that extracts the H&E (Hematoxylin and Eosin) stain pigments and calculates the Laplacian variance to check if the image has the high-frequency texture details expected in microscopy slides.
4. **Prediction Model**: Loaded from a saved Keras model file (.keras or .h5) trained on the IDC (Invasive Ductal Carcinoma) dataset. If no model file is present, the app uses a deterministic mock prediction algorithm for UI testing.

---

## Repository Files

* **app.py**: The main Python script that runs the Flask server, manages page routes, and coordinates the validation and prediction steps.
* **histopath_validate.py**: The validator module that checks if the uploaded file is a valid microscopy slide image before running predictions.
* **Breast_Cancer_Training.ipynb**: The Jupyter Notebook used to train the deep learning model on the breast cancer dataset.
* **breastcancerD.ipynb**: The notebook used for data preprocessing and exploratory data analysis.
* **requirements.txt**: List of Python packages required to run this project.
* **templates/**: Folder containing the HTML pages (Home, Predict, About, and History).
* **static/**: Folder containing the CSS styles and JavaScript logic for the web interface.
* **uploads/**: Folder where uploaded images are temporarily saved for processing.

---

## How to Setup and Run Locally

Follow these steps to run the project on your computer.

### Prerequisites
Make sure you have Python installed on your system. You can download it from python.org.

### Step 1: Clone the repository
Download the code files to your local machine.

### Step 2: Install dependencies
Open your terminal or command prompt, navigate to the project directory, and run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### Step 3: Run the web application
Start the Flask server by running:
```bash
python app.py
```

### Step 4: Open in browser
Once the server starts, open your web browser and go to:
```
http://127.0.0.1:5000
```
You can now upload a sample histopathology image (like the ones included in the root folder, such as `image.jpg`) to test the predictor.

---

## Disclaimer
This project is built as an educational demonstration and coursework project. It is not a certified medical device and should not be used for actual clinical diagnosis or medical decisions.
