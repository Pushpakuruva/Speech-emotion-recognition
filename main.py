import torch
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import wave
import numpy as np
from scipy.io import wavfile
import os

class SpeechEmotionRecognizer:
    def __init__(self, whisper_model="openai/whisper-base", emotion_model="j-hartmann/emotion-english-distilroberta-base"):
        # Initialize Whisper model for speech-to-text
        self.processor = WhisperProcessor.from_pretrained(whisper_model)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
        
        # Initialize emotion classification model
        self.tokenizer = AutoTokenizer.from_pretrained(emotion_model)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model)
        
        # Define emotion labels
        self.emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model.to(self.device)
        self.emotion_model.to(self.device)
        
        print(f"Models loaded successfully. Using device: {self.device}")
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio file using Whisper"""
        # Load and preprocess audio
        try:
            sample_rate, audio = wavfile.read(audio_path)
            # Convert to float32 and normalize
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            
            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                # Simple resampling - for production use a proper resampling library
                audio = np.interp(
                    np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate)),
                    np.arange(0, len(audio)),
                    audio
                )
        except Exception as e:
            print(f"Error loading audio: {e}")
            raise
        
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(input_features)
            
        # Decode the transcription
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
    
    def analyze_emotion(self, text):
        """Classify emotion from text"""
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        # Get emotion predictions
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        # Convert to numpy for processing
        probs = probabilities.cpu().numpy()[0]
        
        # Create dictionary of emotions and their probabilities
        emotions = {self.emotion_labels[i]: float(probs[i]) for i in range(len(self.emotion_labels))}
        
        # Get the dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotions": emotions
        }
    
    def process_audio(self, audio_path):
        """Process audio file and return both transcription and emotion analysis"""
        # Get transcription
        transcription = self.transcribe_audio(audio_path)
        
        # Analyze emotion
        emotion_results = self.analyze_emotion(transcription)
        
        # Return results
        return {
            "transcription": transcription,
            "emotion": emotion_results
        }

    def analyze_audio_segments(self, audio_path, segment_duration=5.0):
        """
        Analyze audio in segments to detect emotion changes throughout the recording
        
        Parameters:
        - audio_path: Path to the audio file
        - segment_duration: Duration of each segment in seconds
        
        Returns:
        - List of dictionaries containing timestamps, transcriptions, and emotions
        """
        # Load audio
        try:
            sample_rate, audio = wavfile.read(audio_path)
            
            # Convert to float32 and normalize
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            
            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                # Simple resampling
                audio = np.interp(
                    np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate)),
                    np.arange(0, len(audio)),
                    audio
                )
                sample_rate = 16000
        except Exception as e:
            print(f"Error loading audio: {e}")
            raise
            
        duration = len(audio) / sample_rate
        
        # Calculate segment samples
        segment_samples = int(segment_duration * sample_rate)
        
        results = []
        
        # Process each segment
        for i in range(0, len(audio), segment_samples):
            segment = audio[i:i+segment_samples]
            
            # Skip segments that are too short
            if len(segment) < 0.5 * segment_samples:
                continue
                
            # Calculate timestamp
            start_time = i / sample_rate
            end_time = min((i + segment_samples) / sample_rate, duration)
            
            # Save segment to temporary file
            temp_path = "temp_segment.wav"
            with wave.open(temp_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                segment_int16 = (segment * 32767).astype(np.int16)
                wf.writeframes(segment_int16.tobytes())
            
            # Process segment
            try:
                transcription = self.transcribe_audio(temp_path)
                
                # Only analyze emotion if transcription isn't empty
                if transcription.strip():
                    emotion_results = self.analyze_emotion(transcription)
                    
                    results.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "transcription": transcription,
                        "emotion": emotion_results
                    })
            except Exception as e:
                print(f"Error processing segment {start_time}-{end_time}: {e}")
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        return results


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--segments", action="store_true", help="Analyze in segments")
    parser.add_argument("--segment_duration", type=float, default=5.0, help="Duration of each segment in seconds")
    args = parser.parse_args()
    
    # Initialize the speech emotion recognizer
    recognizer = SpeechEmotionRecognizer()
    
    if args.segments:
        print(f"Analyzing audio in {args.segment_duration}s segments...")
        segment_results = recognizer.analyze_audio_segments(args.audio, segment_duration=args.segment_duration)
        
        print(f"\nFound {len(segment_results)} segments with speech content")
        for i, segment in enumerate(segment_results):
            print(f"\nSegment {i+1} ({segment['start_time']:.1f}s - {segment['end_time']:.1f}s):")
            print(f"  Transcription: {segment['transcription']}")
            print(f"  Dominant emotion: {segment['emotion']['dominant_emotion']}")
            print("  Top emotions:")
            top_emotions = sorted(segment['emotion']['emotions'].items(), key=lambda x: x[1], reverse=True)[:3]
            for emotion, prob in top_emotions:
                print(f"    {emotion}: {prob:.2f}")
    else:
        print("Processing complete audio file...")
        result = recognizer.process_audio(args.audio)
        
        print(f"\nTranscription: {result['transcription']}")
        print(f"Dominant emotion: {result['emotion']['dominant_emotion']}")
        print("Emotion probabilities:")
        for emotion, prob in sorted(result['emotion']['emotions'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {prob:.4f}")