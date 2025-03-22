from gtts import gTTS
import os

def text_to_speech_hindi(text, output_file="news_audio.mp3"):
    """
    Convert text to Hindi speech and save it as an audio file using gTTS.
    """
    if not text.strip():
        print("⚠️ No text provided for TTS.")
        return
    
    print("🎙️ Generating Hindi speech...")
    tts = gTTS(text=text, lang="hi")
    tts.save(output_file)

    print(f"✅ Audio saved as {output_file}")
    return output_file

# Example usage
if __name__ == "__main__":
    sample_text = "आज की मुख्य खबरें टेस्ला के बारे में हैं।"
    text_to_speech_hindi(sample_text)
