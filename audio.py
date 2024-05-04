from tensorflow.keras.models import load_model
import librosa
import numpy as np

print("Hey")
# Load the saved model
model = load_model("model/audio_classifier.h5")
print(model)
# Define parameters for audio preprocessing (should match parameters used during training)
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109

# Define a function to preprocess audio file


def preprocess_audio(audio_file_path):
    # Load audio file using librosa
    audio, _ = librosa.load(audio_file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Extract Mel spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((
            0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    return mel_spectrogram

# Define a function to predict on audio file


def predict_audio(audio_file_path):
    # Preprocess the audio file
    mel_spectrogram = preprocess_audio(audio_file_path)

    # Reshape the input to match model's input shape
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)

    # Make prediction using the loaded model
    prediction = model.predict(np.array([mel_spectrogram]))

    # Get the predicted class
    predicted_class = "bonafide" if np.argmax(prediction) == 1 else "spoof"

    return predicted_class


# print("Hey")
# audio_path = "Uploaded_Files/" + "0.wav"
# predicted_class = predict_audio(audio_path)
# print("Predicted Class:", predicted_class)