import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import soundfile as sf
import os
import sys

# Load the pretrained YAMNet model
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Load our trained classifier model
classifier_model = tf.keras.models.load_model("final_YamNet_classifier.h5")


# Class labels mapping
class_names = ["Cry", "Scream", "Normal"]

def preprocess_audio(audio_path, output_path="processed_audio.wav", target_sr=16000):
    """
    Converts input audio to WAV format if necessary, resamples to 16 kHz, 16-bit PCM, mono, and normalizes it.
    If input is already a valid WAV file, skips unnecessary processing.
    """
    # Check if file is already a WAV
    if audio_path.lower().endswith(".wav"):
        with sf.SoundFile(audio_path) as f:
            if f.samplerate == target_sr and f.channels == 1 and f.subtype == "PCM_16":
                print("Input is already a valid WAV file. Skipping conversion.")
                return audio_path  # No processing needed

    # Otherwise, process the file
    print("🔄 Processing audio to match required format...")
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None, mono=True)  # Load as mono

    # Convert to 16 kHz if needed
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Normalize the audio (-1 to 1)
    y = y / np.max(np.abs(y))

    # Save as WAV with 16-bit PCM
    sf.write(output_path, y, target_sr, subtype="PCM_16")

    return output_path  # Return processed file path

### **2️⃣ Function to Extract YAMNet Embeddings**
def extract_yamnet_embeddings(audio_path):
    """
    Extracts YAMNet embeddings from an audio file and applies mean pooling.
    """
    # Load audio file (YAMNet expects waveform in range [-1, 1])
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Ensure waveform is a float32 NumPy array
    waveform = np.array(y, dtype=np.float32)

    # Run YAMNet to get embeddings
    _, embeddings, _ = yamnet_model(waveform)

    # Mean pooling to get a (1024,) vector
    mean_embedding = np.mean(embeddings.numpy(), axis=0)

    return mean_embedding  # Shape: (1024,)

### **3️⃣ Function to Classify Audio Using Our Model**
def classify_audio(audio_path):
    """
    Takes an audio file, preprocesses it, extracts embeddings, and predicts the class.
    """
    # Step 1: Preprocess the audio
    processed_audio = preprocess_audio(audio_path)

    # Step 2: Extract mean-pooled YAMNet embeddings
    embedding = extract_yamnet_embeddings(processed_audio)

    # Step 3: Reshape embedding for model input
    embedding = embedding.reshape(1, -1)  # Shape (1, 1024)

    # Step 4: Predict class probabilities
    pred_probs = classifier_model.predict(embedding, verbose=0)

    # Step 5: Get the predicted class
    predicted_class = np.argmax(pred_probs)
    predicted_label = class_names[predicted_class]
    confidence = pred_probs[0][predicted_class]

    # Print final result
   #print(f"Predicted Class: {predicted_label} (Confidence: {confidence:.2f})")

    return predicted_label, confidence


# audio_file = "/home/maditya/Desktop/Front Era/Other/Augmented_Dataset/Normal/normal_09_aug1.wav"
# classify_audio(audio_file)

# Ask the user for the path of the audio file
audio_file = input("Enter the path of the audio file: ")

# Check if the file exists before processing
if not os.path.exists(audio_file):
    print(f"Error: The file '{audio_file}' does not exist.")
    sys.exit(1)  # Exit the script safely

predicted_label = classify_audio(audio_file)
print(f"Predicted Class: {predicted_label}")