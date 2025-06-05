# -*- coding: utf-8 -*-

import os
import numpy as np
from groq import Groq
import librosa
import asyncio
import tempfile
import requests
import whisper
import time
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from pydantic import BaseModel
import uvicorn
from urllib.parse import urlencode
from fastapi.responses import FileResponse, JSONResponse, Response
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq API Client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY", "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"),
)

# ElevenLabs API Key
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_f9bf1a753b40db6ac4391ad23cb9ef2da74de885fd04081c")

# Twilio API Credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "AC0c4f02d6fbb06fb8bc4c25a28cb58ed2")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "7575ad84d043c784546541381ee5f72c")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+16812271336")

# Initialize Twilio Client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Create temp directory for audio files
temp_dir = os.path.join(os.getcwd(), "temp_audio")
os.makedirs(temp_dir, exist_ok=True)

# Use the provided Azure URL or a local URL for development
PUBLIC_URL = os.getenv("WEBSITE_HOSTNAME", "http://localhost:8000")
if not PUBLIC_URL.startswith("http"):
    PUBLIC_URL = f"https://{PUBLIC_URL}"

# Load the Whisper small model
model = whisper.load_model("small")

# FastAPI app
app = FastAPI(title="Dr. Schmidt's Voice Assistant", version="1.0.0")

# Add a heartbeat route to check if service is up
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": str(datetime.now()), "url": PUBLIC_URL}

# Add a simple root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Voice-to-Voice AI Assistant is running", "url": PUBLIC_URL, "timestamp": str(datetime.now())}

# Add a test route to call the assistant directly via web
@app.get("/test")
async def test_view():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dr. Schmidt's Assistant - Test Interface</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .info {{ background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 15px; margin-bottom: 20px; }}
            .btn {{ background-color: #007bff; color: white; border: none; padding: 10px 15px; cursor: pointer; text-decoration: none; }}
            .btn:hover {{ background-color: #0069d9; }}
            .warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dr. Schmidt's Voice Assistant Test Interface</h1>
            <div class="info">
                <p>Public URL: <strong>{PUBLIC_URL}</strong></p>
                <p>Server time: {datetime.now()}</p>
            </div>
            <div class="warning">
                <h3>?? Twilio Free Trial Limitation</h3>
                <p>If you hear "free trial" message, you need to:</p>
                <ul>
                    <li>Upgrade to paid Twilio account, OR</li>
                    <li>Verify your phone number in Twilio Console</li>
                </ul>
            </div>
            <p>To test this assistant:</p>
            <ol>
                <li>Make sure Twilio is configured with the webhook URL: <strong>{PUBLIC_URL}/voice</strong></li>
                <li>Call the phone number: <strong>{TWILIO_PHONE_NUMBER}</strong></li>
                <li>Speak in English or German to schedule an appointment</li>
            </ol>
            <p>
                <a href="{PUBLIC_URL}/voice" target="_blank" class="btn">Test Voice Endpoint</a>
            </p>
        </div>
    </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")

def preprocess_audio(audio_data, sample_rate):
    try:
        # Normalize the audio data
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Apply pre-emphasis
        audio_data = librosa.effects.preemphasis(audio_data, coef=0.97)
        
        return audio_data, sample_rate
    except Exception as e:
        logger.error(f"Error in preprocess_audio: {str(e)}")
        raise

def speech_to_text(audio_url):
    try:
        if not audio_url:
            raise ValueError("No audio URL provided.")
        
        logger.info(f"Downloading audio from URL: {audio_url}")
        # Download audio from URL
        response = requests.get(audio_url, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to download audio: {response.status_code}")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing audio file: {temp_file_path}")
        # Load and process audio
        audio_array, _ = librosa.load(temp_file_path, sr=16000)
        os.remove(temp_file_path)
        
        # Transcribe with language detection limited to English and German
        logger.info("Starting transcription with Whisper...")
        result = model.transcribe(
            audio_array, 
            language=None, 
            task="transcribe", 
            initial_prompt="This is a conversation in English or German about scheduling a medical appointment."
        )
        transcription = result["text"].strip()
        detected_language = result.get("language", "en")
        logger.info(f"Transcription complete: {transcription[:50]}... (Language: {detected_language})")
        
        if detected_language not in ["en", "de"]:
            return "Sorry, I only understand English or German. | Entschuldigung, ich verstehe nur Englisch oder Deutsch.", "en"
        
        return transcription, detected_language
    
    except Exception as e:
        logger.error(f"Error in speech_to_text: {str(e)}")
        return "Sorry, I couldn't understand your speech. Please try again. | Entschuldigung, ich konnte Sie nicht verstehen. Bitte versuchen Sie es erneut.", "en"

def generate_response(text, language="en"):
    try:
        logger.info(f"Generating response for input: {text[:50]}... (Language: {language})")
        
        if language == "de":
            system_prompt = (
                "Sie sind Dr. Schmidts Assistent, ein professioneller menschlicher Assistent, der für die Terminvereinbarung "
                "für Dr. Schmidt zuständig ist. Sie sind kein KI-System. Ihre Aufgabe ist es, Patienten bei der Buchung von "
                "Terminen basierend auf Dr. Schmidts Zeitplan zu helfen. Antworten Sie auf Deutsch und halten Sie die Antworten "
                "kurz und professionell. Erwähnen Sie nicht, dass Sie eine KI sind.\n\n"
                "Dr. Schmidts Terminplan:\n"
                "- Montag bis Freitag: 9:00-12:00 Uhr, 13:00-17:00 Uhr.\n"
                "- Terminslots: Alle 30 Minuten (z.B. 9:00, 9:30, 10:00, etc.).\n"
                "- Nicht verfügbar: Wochenenden, 12:00-13:00 Uhr (Mittagspause).\n\n"
                "Beispielantworten:\n"
                "- \"Guten Tag, ich kann einen Termin für Sie vereinbaren. Wann möchten Sie Dr. Schmidt sehen?\"\n"
                "- \"Welcher Tag und welche Uhrzeit würden Ihnen passen?\""
            )
        else:
            system_prompt = (
                "You are Dr. Schmidt's Assistant, a professional human assistant responsible for scheduling appointments "
                "for Dr. Schmidt. You are not an AI. Your role is to help patients book appointments based on Dr. Schmidt's "
                "schedule. Respond in English and keep responses concise and professional. Do not mention being an AI.\n\n"
                "Dr. Schmidt's schedule:\n"
                "- Monday to Friday: 9:00 AM-12:00 PM, 1:00 PM-5:00 PM.\n"
                "- Appointment slots: Every 30 minutes (e.g., 9:00, 9:30, 10:00, etc.).\n"
                "- Unavailable: Weekends, 12:00 PM-1:00 PM (lunch).\n\n"
                "Example responses:\n"
                "- \"Hello, I can book an appointment for you. When would you like to see Dr. Schmidt?\"\n"
                "- \"What day and time would work best for you?\""
            )
        
        logger.info("Calling Groq API...")
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=150
        )
        response = chat_completion.choices[0].message.content.strip()
        logger.info(f"Received response from Groq: {response[:50]}...")
        return response
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        if language == "de":
            return "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut."
        else:
            return "Sorry, an error occurred. Please try again."

async def text_to_speech(text, language="en"):
    try:
        logger.info(f"Converting to speech: {text[:50]}... (Language: {language})")
        output_path = os.path.join(temp_dir, f"response_{os.getpid()}_{int(time.time())}.mp3")
        
        # Choose appropriate voice based on language
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default English voice
        if language == "de":
            voice_id = "pNInz6obpgDQGcFmaJgB"  # German voice (Adam)
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.6,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        logger.info("Calling ElevenLabs API...")
        response = requests.post(url, json=data, headers=headers, timeout=30)
        if response.status_code != 200:
            raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Generated audio file not found at {output_path}")
        
        logger.info(f"Speech generated successfully: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        return None

async def process_call(audio_url, language=None):
    logger.info(f"Processing call with audio URL: {audio_url}")
    try:
        text_input, detected_language = await asyncio.to_thread(speech_to_text, audio_url)
        logger.info(f"Transcription: {text_input} (Language: {detected_language})")
        
        # Handle error messages from speech_to_text
        if "Sorry, I only understand" in text_input or "Entschuldigung" in text_input:
            response_audio_path = await text_to_speech(text_input, "en")
            return text_input, response_audio_path
        
        response_text = await asyncio.to_thread(generate_response, text_input, detected_language)
        logger.info(f"Response text: {response_text[:50]}...")
        
        response_audio_path = await text_to_speech(response_text, detected_language)
        logger.info(f"Generated audio at: {response_audio_path}")
        return response_text, response_audio_path
    
    except Exception as e:
        logger.error(f"General error in process_call: {str(e)}")
        error_msg = "Sorry, there was a technical issue. Please try again. | Entschuldigung, es gab ein technisches Problem. Bitte versuchen Sie es erneut."
        return error_msg, None

# Handle voice requests with common function for both GET and POST
async def handle_voice_common():
    logger.info(f"[{datetime.now()}] Handling voice request")
    response = VoiceResponse()
    
    # Gather speech input with improved settings
    gather = response.gather(
        input="speech",
        action=f"{PUBLIC_URL}/process_speech",
        method="POST",
        speech_timeout="auto",
        language="en-US,de-DE",  # Support English and German
        timeout=10,
        finish_on_key="#",
        speech_model="experimental_conversations"  # Better speech recognition
    )
    
    # Bilingual greeting
    gather.say(
        "Hello, welcome to Dr. Schmidt's appointment assistant. Please tell me when you would like to schedule your appointment. "
        "Hallo, willkommen bei Dr. Schmidts Terminassistent. Bitte sagen Sie mir, wann Sie Ihren Termin vereinbaren möchten.",
        voice="alice",
        language="en-US"
    )
    
    # Add a fallback in case the user doesn't speak
    response.say("I didn't hear anything. Please call back when you're ready to schedule an appointment.")
    
    logger.info(f"Voice response: {str(response)[:100]}...")
    return Response(content=str(response), media_type="application/xml")

@app.get("/voice")
async def handle_voice_get():
    logger.info(f"[{datetime.now()}] GET request to /voice")
    return await handle_voice_common()

@app.post("/voice")
async def handle_voice_post(request: Request):
    logger.info(f"[{datetime.now()}] POST request to /voice")
    form_data = await request.form()
    logger.info(f"Form data: {dict(form_data)}")
    return await handle_voice_common()

@app.get("/process_speech")
async def process_speech_get(request: Request):
    logger.info(f"[{datetime.now()}] GET request to /process_speech - this endpoint should be accessed via POST")
    response = VoiceResponse()
    response.say("This endpoint requires a POST request with speech data.")
    return Response(content=str(response), media_type="application/xml")

@app.post("/process_speech")
async def process_speech(request: Request):
    logger.info(f"[{datetime.now()}] Processing speech request")
    form_data = await request.form()
    logger.info(f"Received form data: {dict(form_data)}")
    
    # Get speech result or recording URL
    speech_result = form_data.get("SpeechResult")
    recording_url = form_data.get("RecordingUrl")
    
    response = VoiceResponse()
    
    # Handle speech recognition result
    if speech_result:
        logger.info(f"Speech result received: {speech_result}")
        # Process text directly
        detected_language = "de" if any(word in speech_result.lower() for word in ["termin", "ich", "möchte", "wann"]) else "en"
        response_text = await asyncio.to_thread(generate_response, speech_result, detected_language)
        response_audio_path = await text_to_speech(response_text, detected_language)
        
        if response_audio_path:
            audio_url = f"{PUBLIC_URL}/audio/{os.path.basename(response_audio_path)}"
            logger.info(f"Playing response audio from: {audio_url}")
            response.play(audio_url)
        else:
            response.say(response_text, voice="alice")
    
    # Handle recording URL
    elif recording_url:
        logger.info(f"Processing audio from URL: {recording_url}")
        response_text, response_audio_path = await process_call(recording_url)
        
        if response_audio_path:
            audio_url = f"{PUBLIC_URL}/audio/{os.path.basename(response_audio_path)}"
            logger.info(f"Playing response audio from: {audio_url}")
            response.play(audio_url)
        else:
            logger.info(f"No audio path returned, using text-to-speech: {response_text}")
            response.say(response_text or "Sorry, an error occurred. Please try again.", voice="alice")
    
    else:
        logger.warning("No speech result or recording URL found in request")
        response.say("Sorry, I didn't hear anything. Please try again.", voice="alice")
    
    # Gather more input for continued conversation
    gather = response.gather(
        input="speech",
        action=f"{PUBLIC_URL}/process_speech",
        method="POST",
        speech_timeout="auto",
        language="en-US,de-DE",
        timeout=10,
        finish_on_key="#"
    )
    gather.say("Is there anything else I can help you with?", voice="alice")
    
    # Final goodbye
    response.say("Thank you for calling. Goodbye!", voice="alice")
    
    logger.info(f"Response TwiML: {str(response)[:200]}...")
    return Response(content=str(response), media_type="application/xml")

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    file_path = os.path.join(temp_dir, filename)
    logger.info(f"[{datetime.now()}] Serving audio file: {file_path}")
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            media_type="audio/mpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    logger.error(f"Audio file not found: {file_path}")
    raise HTTPException(status_code=404, detail="Audio file not found")

# Update Twilio webhook URL
def update_twilio_webhook(url):
    try:
        logger.info(f"Updating Twilio webhook URL to: {url}")
        # Get the phone number
        numbers = twilio_client.incoming_phone_numbers.list(phone_number=TWILIO_PHONE_NUMBER)
        if not numbers:
            logger.error(f"Phone number {TWILIO_PHONE_NUMBER} not found in account")
            return False
            
        number = numbers[0]
        # Update the voice URL
        number.update(voice_url=url, voice_method='POST')
        logger.info(f"Successfully updated Twilio webhook for {TWILIO_PHONE_NUMBER}")
        return True
    except Exception as e:
        logger.error(f"Error updating Twilio webhook: {str(e)}")
        return False

# Cleanup old audio files
async def cleanup_old_files():
    try:
        current_time = time.time()
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 3600:  # Delete files older than 1 hour
                    os.remove(file_path)
                    logger.info(f"Deleted old audio file: {filename}")
    except Exception as e:
        logger.error(f"Error cleaning up old files: {str(e)}")

if __name__ == "__main__":
    # Configure Twilio webhook if we're not in development mode
    if PUBLIC_URL != "http://localhost:8000":
        webhook_url = f"{PUBLIC_URL}/voice"
        logger.info(f"Setting Twilio webhook to: {webhook_url}")
        
        # Automatically update Twilio webhook
        if update_twilio_webhook(webhook_url):
            logger.info("Twilio webhook updated successfully")
        else:
            logger.error("Failed to update Twilio webhook automatically")
            logger.info("Please manually configure your Twilio phone number with this webhook URL using POST method")
    
    # Run FastAPI server (this will be used only for local development)
    # In Azure, Gunicorn will serve the app instead
    logger.info(f"Starting server at {datetime.now()}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))