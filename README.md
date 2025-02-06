# AI_Group17



# **Fashion-MNIST Image Classification with Flask & Docker Deployment**
🚀 **A deep learning-based image classification system using CNN and Flask, with an attempt to deploy using Docker.**  

---

## **📌 Project Overview**
This project implements a **Convolutional Neural Network (CNN)** trained on the **Fashion-MNIST dataset** to classify clothing items into **10 categories**.  
It features:  
✔️ **A Flask Web Interface** for testing image classification.  
✔️ **A locally deployed UI** for users to upload images and get predictions.  
✔️ **An attempted Docker Deployment** (troubleshooting required).  

---

## **📂 Directory Structure**
```
fashion-mnist-classifier/
│── fashion/                    # Dataset folder (CSV train & test files)
│── fashion_mnist_model.h5       # Trained CNN model
│── app.py                        # Flask API
│── templates/                    # HTML templates for web interface
│   ├── index.html                 # Upload form
│   └── result.html                # Prediction result
│── static/                        # Static files (CSS, JS, images)
    ├── style.css    
│── requirements.txt               # Dependencies
│── Dockerfile                     # Docker configuration (not fully working)
│── README.md                      # Project documentation
```

---

## **📌 Dataset**
- **Fashion-MNIST Dataset** from Zalando, consisting of **60,000 training images** and **10,000 test images** in **10 classes**:
  1. T-shirt/top  
  2. Trouser  
  3. Pullover  
  4. Dress  
  5. Coat  
  6. Sandal  
  7. Shirt  
  8. Sneaker  
  9. Bag  
  10. Ankle boot

- Stored in **CSV format** inside the `fashion/` directory.

---

## **🛠 Installation & Running Locally**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/vicky3663/AI_Group17
cd AI_Group17
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Flask App**
```bash
python app.py
```
- Open `http://127.0.0.1:5000/` in your browser.
- Upload an image and see the prediction.

---

## **📌 Model Training**
To train the CNN model manually:
```python
python train.py
```
- This will save the trained model as `fashion_mnist_model.h5`.

---

## **📌 Flask UI Setup**
The web interface allows users to **upload an image** and receive **real-time predictions**.

- **Frontend (HTML Form)**: Located in the `templates/index.html` file.
- **Backend (Flask API)**:
  - Loads the trained model.
  - Accepts uploaded images.
  - Preprocesses the image and predicts the clothing category.

---

## **🚀 Attempted Docker Deployment**
### **1️⃣ Build the Docker Image**
```bash
docker build -t fashion-mnist-app .
```

### **2️⃣ Run the Docker Container**
```bash
docker run -p 5000:5000 fashion-mnist-app
```

#### **💡 Issues Faced**
- If the container **fails to run**, check logs using:
  ```bash
  docker logs <container_id>
  ```
- Common problems:
  - Missing dependencies inside the Docker container.
  - TensorFlow version conflicts.
  - Flask not binding correctly (`host='0.0.0.0'` may be needed in `app.py`).

---

## **📌 Deployment Troubleshooting**
- Ensure `Dockerfile` installs all dependencies (`tensorflow`, `flask`, etc.).
- Verify that Flask is set to run on **host=0.0.0.0** and the correct port (`5000`).
- Run the container interactively to debug:
  ```bash
  docker run -it fashion-mnist-app /bin/bash
  ```
- If needed, try running without GPU support:
  ```bash
  docker run --rm -p 5000:5000 --env TF_CPP_MIN_LOG_LEVEL=2 fashion-mnist-app
  ```

---

## **📜 License**
This project is open-source under the **MIT License**.

---

### **Next Steps**
✅ Fix Docker Deployment  
✅ Improve UI for better user experience  
✅ Optimize model performance  

