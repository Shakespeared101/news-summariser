import edge_tts
import asyncio

async def text_to_speech_hindi(text, output_file="news_audio.mp3"):
    """
    Convert text to Hindi speech and save as an audio file using Edge TTS.
    """
    if not text.strip():
        print("⚠️ No text provided for TTS.")
        return

    print("🎙️ Generating Hindi speech...")
    communicate = edge_tts.Communicate(text, voice="hi-IN-MadhurNeural")
    await communicate.save(output_file)

    print(f"✅ Audio saved as {output_file}")
    return output_file

# Example usage
if __name__ == "__main__":
    asyncio.run(text_to_speech_hindi("आज की मुख्य खबरें टेस्ला के बारे में हैं।"))
