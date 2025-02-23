import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import gc

# Class labels
class_names = ["cry", "scream", "normal"]

# Function to load and run YAMNet inference
def predict_probs_yamnet(audio_file):
    """ Load YAMNet, get probabilities, then delete the model to free memory. """
    
    # Load the pretrained YAMNet model
    with tf.device('/CPU:0'):  # Force CPU for TensorFlow
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    # Load trained classifier model
    classifier_model = tf.keras.models.load_model("final_YamNet_classifier.h5")

    # Preprocess audio
    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    waveform = np.array(y, dtype=np.float32)

    # Extract embeddings
    _, embeddings, _ = yamnet_model(waveform)
    mean_embedding = np.mean(embeddings.numpy(), axis=0).reshape(1, -1)

    # Get probabilities
    pred_probs = classifier_model.predict(mean_embedding, verbose=0)[0]

    # **Delete model to free memory**
    del yamnet_model, classifier_model
    tf.keras.backend.clear_session()
    gc.collect()

    return pred_probs  # Shape (3,)

# Define FFNN classifier for Wav2Vec2 embeddings
class FFNN(nn.Module):
    def __init__(self, input_dim=768, dropout_rate=0.3):
        """
        A feed-forward neural network with Dropout and BatchNorm for regularization.
        """
        super(FFNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)  
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)  
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.fc5 = nn.Linear(128, 3)  # Output for 3 classes

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = self.fc5(x)  # Logits for classification
        return x

# Function to load and run Wav2Vec2 inference
def predict_probs_wav2vec2(audio_file):
    """ Load Wav2Vec2, get probabilities, then delete the model to free memory. """

    # Path to locally downloaded Wav2Vec2
    base_path = "./models/wav2vec2/models--facebook--wav2vec2-base-960h/snapshots"
    snapshot_folder = os.listdir(base_path)[0]  # Get latest downloaded snapshot
    local_model_path = os.path.join(base_path, snapshot_folder)

    # Load Wav2Vec2 model and feature extractor
    wav2vec2_model = Wav2Vec2Model.from_pretrained(local_model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local_model_path)
    
    wav2vec2_model.eval()
    wav2vec2_model.to("cpu")

    # Preprocess audio
    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")

    # Extract embeddings
    with torch.no_grad():
        outputs = wav2vec2_model(**inputs)
    mean_emb = outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()

    # Load trained FFNN classifier
    device = torch.device("cpu")
    acoustic_classifier = FFNN(input_dim=768, dropout_rate=0.3).to(device)
    model_path = "best_model.pth"
    acoustic_classifier.load_state_dict(torch.load(model_path, map_location=device))
    acoustic_classifier.eval()

    # Get logits and convert to probabilities
    tensor_emb = torch.from_numpy(mean_emb).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = acoustic_classifier(tensor_emb)
    pred_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # **Delete model to free memory**
    del wav2vec2_model, feature_extractor, acoustic_classifier, tensor_emb, logits
    gc.collect()
    torch.cuda.empty_cache()

    return pred_probs  # Shape (3,)

# Ask for audio file path
audio_file = input("Enter the path of the audio file: ")

if not os.path.exists(audio_file):
    print(f"Error: The file '{audio_file}' does not exist.")
    sys.exit(1)

# **Run models one after the other**
p_yam = predict_probs_yamnet(audio_file)  # Get YAMNet probabilities
p_wav = predict_probs_wav2vec2(audio_file)  # Get Wav2Vec2 probabilities

# **Ensemble Methods**
def ensemble_average(p_yam, p_wav):
    p_ensemble = (p_yam + p_wav) / 2.0
    final_class_idx = np.argmax(p_ensemble)
    return class_names[final_class_idx], p_ensemble[final_class_idx]

def ensemble_weighted(p_yam, p_wav, alpha=0.6):
    p_ensemble = alpha * p_yam + (1 - alpha) * p_wav
    final_class_idx = np.argmax(p_ensemble)
    return class_names[final_class_idx], p_ensemble[final_class_idx]

# Choose ensemble method
choice = input("Choose Ensemble Method: [1] Average, [2] Weighted: ")

if choice == "1":
    predicted_label, confidence = ensemble_average(p_yam, p_wav)
elif choice == "2":
    alpha = float(input("Enter alpha (e.g., 0.6 for 60% YAMNet, 40% Wav2Vec2): "))
    predicted_label, confidence = ensemble_weighted(p_yam, p_wav, alpha)
else:
    print("Invalid choice! Defaulting to Averaging.")
    predicted_label, confidence = ensemble_average(p_yam, p_wav)

print(f"Final Prediction: {predicted_label} (Confidence: {confidence:.2f})")