import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import os
import librosa
import tensorflow as tf
import tensorflow_hub as hub  # or your TF-based Wav2Vec2
import torch.nn as nn
import sys

# Load the pretrained Wav2Vec2 model
# Load Wav2Vec2 from local storage
# local_wav2vec_path = "models/wav2vec2/"  # Assumes the model is stored here
# wav2vec2_model = Wav2Vec2Model.from_pretrained(local_wav2vec_path)
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local_wav2vec_path)

base_path = "./models/wav2vec2/models--facebook--wav2vec2-base-960h/snapshots"
snapshot_folder = os.listdir(base_path)[0]  # Get the first folder inside snapshots

local_model_path = os.path.join(base_path, snapshot_folder)  # Final model path

# Load the model
wav2vec2_model = Wav2Vec2Model.from_pretrained(local_model_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local_model_path)


wav2vec2_model.eval()      # Inference mode
wav2vec2_model.to("cpu")   # or "cuda" if you have a GPU

def preprocess_audio(input_path, output_path="temp_processed.wav", target_sr=16000):
    """
    Converts input audio to WAV (16kHz, 16-bit PCM, mono).
    If it's already valid, you could skip re-saving. Here we always ensure consistency.
    """
    # Load with librosa (auto-resamples if sr != None).
    y, sr = librosa.load(input_path, sr=None, mono=True)

    # If sample rate != target_sr, resample
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Normalize to -1..1
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Save as 16-bit PCM WAV
    sf.write(output_path, y, target_sr, subtype="PCM_16")

    return output_path

def extract_wav2vec2_embedding(audio_path):
    """
    Loads audio at 16kHz, uses Wav2Vec2FeatureExtractor & Wav2Vec2Model to get (hidden_size,) embedding.
    """
    # 1) Load audio at 16kHz
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # 2) Convert waveform to Wav2Vec2 inputs
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")

    # 3) Forward pass
    with torch.no_grad():
        outputs = wav2vec2_model(**inputs)  # last_hidden_state shape: (1, seq_len, hidden_size)

    # 4) Squeeze batch dimension => (seq_len, hidden_size)
    last_hidden = outputs.last_hidden_state.squeeze(0)

    # 5) Mean-pool over time => shape (hidden_size,)
    mean_emb = last_hidden.mean(dim=0).cpu().numpy()

    return mean_emb

class FFNN(nn.Module):
    def __init__(self, input_dim=768, dropout_rate=0.3):
        """
        A feed-forward neural network with Dropout and BatchNorm for regularization.
        Adjust input_dim if your embeddings are not 768 in size.
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



# Load your trained weights
device = torch.device("cpu")  # or "cuda"
acoustic_classifier = FFNN(input_dim=768, dropout_rate=0.3).to(device)
model_path = "best_model.pth"  # Load from the same folder as the script
acoustic_classifier.load_state_dict(torch.load(model_path, map_location=device))
acoustic_classifier.eval()

def classify_audio(input_audio_path):
    """
    1) Preprocess to 16kHz WAV
    2) Extract Wav2Vec2 embedding
    3) Pass embedding to PyTorch FFNN
    4) Print & return predicted label
    """
    # Step 1) Preprocess audio
    processed_path = preprocess_audio(input_audio_path, "temp_processed.wav")

    # Step 2) Extract Wav2Vec2 embedding => shape (768,)
    embedding = extract_wav2vec2_embedding(processed_path)

    # Convert to torch tensor => shape (1, 768)
    tensor_emb = torch.from_numpy(embedding).unsqueeze(0).float().to(device)

    # Step 3) Classify with FFNN
    with torch.no_grad():
        logits = acoustic_classifier(tensor_emb)  # shape (1, 3)
        pred_idx = torch.argmax(logits, dim=1).item()

    # Step 4) Map predicted index to label
    label_map = {0: "cry", 1: "scream", 2: "normal"}
    predicted_label = label_map[pred_idx]

    #print(f"Predicted Class: {predicted_label}")
    return predicted_label


# audio_file = "/home/maditya/Desktop/Front Era/Model2 :- Wav2Vec2/Final_Dataset_with_aug/Cry/cry_002_aug.wav"
# predicted_label = classify_audio(audio_file)

# Ask the user for the path of the audio file
audio_file = input("Enter the path of the audio file: ")

# Check if the file exists before processing
if not os.path.exists(audio_file):
    print(f"Error: The file '{audio_file}' does not exist.")
    sys.exit(1)  # Exit the script safely

predicted_label = classify_audio(audio_file)
print(f"Predicted Class: {predicted_label}")