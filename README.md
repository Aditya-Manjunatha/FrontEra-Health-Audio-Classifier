# 🎙️ Developing an Ensemble Model for Detecting Infant Cries, Screams, and Normal Utterances  

## 📌 Project Overview  

The goal of this project is to **classify audio files into one of three categories**:  
1️⃣ **Cry**  
2️⃣ **Scream**  
3️⃣ **Normal Speech**  

To accomplish this task, we **fine-tune two pretrained models** and create an **ensemble model** for improved performance:  

### 🔹 **Pretrained Models Used:**  
✅ **[YAMNet](https://www.researchgate.net/figure/YAMNet-Body-Architecture-Conv-Convolution-dw-Depthwise-pw-Pointwise_fig2_354549303)** - A deep neural network for sound classification trained on AudioSet.  
✅ **[Wav2Vec2](https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html)** - A self-supervised model for speech and sound processing.  

Once fine-tuned, we create **three inference pipelines**:  
1️⃣ **YAMNet-based classifier**  
2️⃣ **Wav2Vec2-based classifier**  
3️⃣ **Ensemble model** combining both predictions  



## 📌 Setting Up the Environment  

Before diving into data preprocessing and model training, we need to **set up the environment** to ensure smooth execution of the code.  



### **1. Create a Virtual Environment**  
It is **recommended** to use a virtual environment to keep dependencies organized and avoid conflicts.  

#### **For Linux & macOS:**  
```bash
python3 -m venv env
source env/bin/activate
```
## **📌 Setting Up the Repository & Organizing Files**  

Before running inference, follow these steps to **set up the repository, download necessary files, and organize them properly**.

---

### **1️⃣ Clone the Repository & Set Up the Folder**  
Run the following commands to create an `AudioClassifier` folder and clone the repo inside it:  

```bash
mkdir AudioClassifier && cd AudioClassifier

# Clone the repository
git clone https://github.com/Aditya-Manjunatha/FrontEra-Health-Audio-Classifier.git .

# Verify that required files are in place
ls -l
```
### **2️⃣ Download & Store Wav2Vec2 Locally**  
To avoid downloading the Wav2Vec2 model every time, store it inside the folder :
```bash
mkdir models
huggingface-cli download facebook/wav2vec2-base-960h --cache-dir ./models/wav2vec2/

```
### **3️⃣ Install Dependencies** 

A requirements.txt file is provided to install the libraries to be installed

```bash
pip install -r requirements.txt
```
### **4️⃣ Move your Audio Files to Folder** 
If you already have a folder containing your test audio files, move it into AudioClassifier/:
```bash
mv /path/to/your/test_audio/ AudioClassifier/
```
If you don’t have a test_audio folder, create it manually and add your test .wav files inside:
```bash
mkdir AudioClassifier/test_audio
mv /path/to/your_audio.wav AudioClassifier/test_audio/
```

### **5️⃣ Verify Folder Structure**
Verify that the folder looks like this now
```bash
AudioClassifier/
│── YamNet_infer.py
│── Wav2Vec2_infer.py
│── Ensemble_infer.py
│── final_YamNet_classifier.h5
│── best_model.pth
│── requirements.txt
│── report.pdf
│── models/
│   └── wav2vec2/  # Locally downloaded Wav2Vec2 model
│── test_audio/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
│── Model_Training_Notebooks/  # Jupyter notebooks used for training
│   ├── YamNet_Training.ipynb
│   ├── Wav2Vec2_Training.ipynb
│   ├── Data_Preprocessing.ipynb
│── Datasets/  # Data used for training
│   ├── Cry/
│   │   ├── cry_01.wav
│   │   ├── cry_02_aug.wav
│   │   └── ...
│   ├── Scream/
│   │   ├── scream_01.wav
│   │   ├── scream_02_aug.wav
│   │   └── ...
│   ├── Normal/
│   │   ├── normal_01.wav
│   │   ├── normal_02_aug.wav
│   │   └── ...
```



## **4. Running Inference on an Audio File**  

Once the **model weights** and **inference scripts** are in place, you can **run inference on any test audio file** to classify it as **Cry, Scream, or Normal**.

**NOTE :- Ensure you have cd'd AudioClassifier Folder**
### **🔹 Step 1: Run an Inference Script**  
After placing the inference scripts and model weights in the correct directory, choose which model to run and execute the corresponding script.

#### **Run YAMNet Inference**  
```bash
python YamNet_infer.py
```
#### **Run Wav2Vec2 Inference**  
```bash
python Wav2Vec2_infer.py
```
#### **Run Ensemble Inference**  
```bash
python Ensemble_infer.py
```

### **🔹 Step 3: Provide the Audio File Path**  
Once you run the script for `YamNet_infer.py` OR `Wav2Vec2_infer.py`, it will prompt you to enter the path of an audio file.

Example :
```bash
Enter the path of the audio file: /test_audio/sample1.wav

```

If you run the script `Ensemble_infer.py` it will ask you which type of ensemble method you want . 

Example :
```bash
Choose Ensemble Method:
1 - Averaging
2 - Weighted Averaging (default weight: α = 0.7)
Enter your choice (1 or 2): 1
```
Once you select the method, it will ask for the audio file path as seen before

### **🔹 Step 4: Get the Prediction** 
Wait for the model to process the audio file and return the predicted label.

Example Output (YAMNet):

```bash
🔄 Processing audio to match required format...
✅ Extracting YAMNet embeddings...
✅ Running classification model...
Predicted Class: Scream (Confidence: 0.94)
```

Example Output (Ensemble Model - Average & Weighted Predictions):

```bash
🔄 Processing audio to match required format...
✅ Extracting YAMNet embeddings...
✅ Extracting Wav2Vec2 embeddings...
✅ Running classification models...

🔹 Ensemble (Averaging) Prediction => Scream (Confidence: 0.92)
🔹 Ensemble (Weighted α=0.7) Prediction => Scream (Confidence: 0.95)

🟢 Final Predictions:
✅ Ensemble (Averaging) → Scream
✅ Ensemble (Weighted) → Scream

```
---
## **5. Diving Deeper - Data**  

Now that the setup is complete, we can explore the **data collection, preprocessing, and augmentation strategies** used in this project.  



### **🔹 Data Collection**  
We gathered audio data from multiple sources to ensure a diverse and balanced dataset:  

1️⃣ **[Infant Cry Audio Corpus](https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus)** – Contains various infant cry recordings.  
2️⃣ **[Infant's Cry Sound Dataset](https://data.mendeley.com/datasets/hbppd883sd/1)** – Another dataset featuring different infant cries.  
3️⃣ **[Human Screaming Detection Dataset](https://www.kaggle.com/datasets/whats2000/human-screaming-detection-dataset)** – Includes human screams from different environments.  
4️⃣ **[Common Voice Dataset](https://www.kaggle.com/datasets/mozillaorg/common-voice)** – Provides normal human speech recordings.  



### **🔹 Audio Preprocessing**  
Upon reviewing the **YAMNet** and **Wav2Vec2** documentation, I identified the required **audio specifications**:  

✅ **WAV Format** 

✅ **16 kHz sample rate**  

✅ **16-bit PCM encoding**  

✅ **Mono-channel audio**  

All collected audio files were **preprocessed accordingly** using **Librosa** and **SoundFile**, ensuring compatibility with the models. Preprocessing steps included:  

- **Resampling** audio to **16 kHz** (if different).  
- **Converting to PCM16** format.  
- **Ensuring mono-channel audio** for consistency.  



### **🔹 Data Augmentation**  

Since our dataset had **limited samples**, we applied **data augmentation** techniques to generate additional training examples and improve model generalization.  

📌 **Class Distribution Before Augmentation:**  
- 🍼 **Cry Class** → **565 samples**  
- 📢 **Scream Class** → **862 samples**  
- 🗣️ **Normal Utterance Class** → **4,067 samples** (sampled **~800**)  

To balance the dataset, we randomly applied one of the following augmentations to each sample to **create more datapoints** 

✅ **Time Stretching** – Slowing down or speeding up the audio without changing pitch.  
✅ **Pitch Shifting** – Raising or lowering the pitch while maintaining tempo.  
✅ **Background Noise Addition** – Adding white noise or random environmental sounds.  
✅ **Volume Perturbation** – Slightly increasing or decreasing audio loudness.  
✅ **Time Shifting** – Introducing slight delays to modify the starting point.  

These augmentations **helped improve model robustness** by exposing it to **varied audio conditions** while maintaining the core characteristics of the target classes.  


### **🔹 Dataset Readiness**  
After preprocessing and augmentation, our dataset was **fully prepared** after combining both the original and the augmented datapoints. This dataset is then used for **fine tuning** **YAMNet** and **Wav2Vec2**.

## **5.1. Diving Deeper - Fine-Tuning**  

### **🔹 What is Fine-Tuning?**  
Fine-tuning is the process of **adapting a pretrained model** to a **specific task** by training it on a smaller, task-specific dataset. There are **two common approaches** to fine-tuning:  

1️⃣ **Changing the classifier head** – Replacing the final classification layer while keeping the pretrained feature extractor **frozen**.  
2️⃣ **Unfreezing layers & training** – Unfreezing some (or all) layers and training them on new data.  

Since our dataset was **limited in size**, and we had **compute constraints**, we chose the **first approach**—using pretrained models as **feature extractors** and training our own classifier on the extracted embeddings.  



## **📌 5.1.1 Fine-Tuning YAMNet**  

### **🔹 Understanding YAMNet’s Architecture**  
YAMNet is a **deep audio classification model** trained on **AudioSet**, capable of recognizing **521 audio event classes**. Instead of modifying its classifier, we **leveraged its embeddings**:  

📌 **Key Observation:**  
- In YAMNet, the **second-to-last layer** outputs a **1024-dimensional embedding** for each audio frame.  
- These embeddings are then passed into a **512-class classifier** in the original model.  

### **🔹 Extracting Embeddings**  
To **adapt YAMNet to our task**, we:  
✅ Passed each audio file through YAMNet to **extract embeddings**.  
✅ Applied **mean pooling** to aggregate frame-level embeddings into a **single 1024-dimensional vector per file**.  
✅ Used these **mean-pooled embeddings** as input features for our classifier.  

### **🔹 Custom FFNN Classifier**  
We designed a **Feedforward Neural Network (FFNN)** to classify the extracted YAMNet embeddings.  

🔹 **FFNN Architecture:**  
- **Input Layer** → `1024 neurons` (YAMNet embedding size)  
- **Hidden Layers** → `3 layers` with **ReLU activation** and **dropout** for regularization  
- **Output Layer** → `3 neurons` (Softmax activation for `Cry`, `Scream`, `Normal`)  

## **📌 5.1.2 Fine-Tuning Wav2Vec2**  

### **🔹 Understanding Wav2Vec2’s Architecture**  
Wav2Vec2 is a **self-supervised speech model** trained on large-scale speech datasets. It consists of:  

🔹 **Key Components:**  
- **Feature Encoder** → Convolutional layers that extract raw audio features.  
- **Transformer Encoder** → Outputs a `768-dimensional` feature vector for each audio frame.  
- **Final Classification Head** (which we **ignored**).  


### **🔹 Extracting Embeddings**  
To **adapt Wav2Vec2 to our task**, we:  
✅ Passed each audio file through **Wav2Vec2’s transformer encoder** to **extract 768-dimensional embeddings per frame**.  
✅ Applied **mean pooling** to obtain a **single 768-dimensional vector per file**.  
✅ Used these **mean-pooled embeddings** as input features for our classifier.  


### **🔹 Custom FFNN Classifier**  
Similar to YAMNet, we designed a **FFNN classifier** for the Wav2Vec2 embeddings.  

🔹 **FFNN Architecture:**  
- **Input Layer** → `768 neurons` (Wav2Vec2 embedding size)  
- **Hidden Layers** → `3 layers` with **BatchNorm, ReLU activation, and Dropout**  
- **Output Layer** → `3 neurons` (Softmax activation for `Cry`, `Scream`, `Normal`)  

## **📌 5.2 Diving Deeper - Training and Hyperparameter Tuning**  

### **🔹 Hyperparameter Tuning with K-Fold Cross-Validation**  
To select the optimal hyperparameters, we initialized **5 different hyperparameter combinations** and performed **K-Fold Cross-Validation** on each.  

✅ **K-Fold Cross-Validation** was used to ensure that the model **generalizes well** and does not overfit.  
✅ For each fold, we **trained and validated** on different splits and **computed key metrics** like **accuracy, precision, recall, and F1-score**.  
✅ The **detailed results** of each hyperparameter combination are provided in the **REPORT**.  



### **🔹 Selecting the Best Hyperparameters**  
After evaluating the **average accuracy across all folds**, we selected the **best-performing hyperparameter combination** and used it to train the final model.  

📌 The **best hyperparameter combination** was then used in the **main training loop** to train the model on the **entire 70% training set**.  



### **🔹 Final Model Training**  
Once the best hyperparameters were chosen:  
✅ The model was trained on the **full training set (70%)**.  
✅ Metrics such as **training accuracy, validation accuracy, and loss curves** were monitored to prevent overfitting.  
✅ The **final trained model** was saved for inference and testing.  



### **🔹 Final Model Testing**  
After training, the model was evaluated on the **held-out test set (15%)**.  
✅ **Test accuracy, precision, recall, and F1-score** were computed.  
✅ A **detailed performance analysis** with visualizations like **confusion matrices and ROC curves** is provided in the **REPORT**.  

---






