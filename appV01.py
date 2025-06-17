# 1. imports
from dotenv import load_dotenv
import os
import json
from mem0 import MemoryClient, Memory
from google import genai

# Intializations
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
mem_api_key = os.getenv('MEM_API_KEY')

mem_memoryClient = MemoryClient(api_key=mem_api_key)
google_client = genai.Client(api_key=google_api_key)

def google_image_desc_context(file_path):
    sys_prompt = "You a are very talented stroyteller who can create a amazing, attention grabing stories just by looking at the picture, Your main task as a storyteller is to give best context for a story so that another model can generate seemlessly"
    sys_prompt += "Your main 2 task is to give 1. complete discription of the image 2. give the base context as well as ending context for the story."
    sys_prompt += "You will always return the response in the json format:"
    sys_prompt += """
{
    "story_context": 'here we will have the context for the story',
    "image_description": "here we will have the complete description for the story"
}    
"""

    my_file = google_client.files.upload(file=file_path)

    res = google_client.models.generate_content(
        model='gemini-2.0-flash',
        contents = [my_file, sys_prompt],
        config={
        "response_mime_type": "application/json",
        # "response_schema": list[Recipe],
        },
    )

    return json.loads(res.text)

json_data = google_image_desc_context("/workspaces/ChatBot_with_Long_Term_Memory/AirlineAssisntantAI.png")
print(json_data['image_description'])
print(json_data['story_context'])

def story_teller(context):
    res = google_client.models.generate_content(
        model = 'gemini-2.0-flash',
        contents = f"You are a very good story teller, who can create a attention grabing story just with the use of context, Your job is to create a story based on the provived context as this {context}, create a story within 150 words"
    )
    return res.text
print("---"*50)
print(story_teller(json_data['story_context']))

from google import genai
from google.genai import types
import wave

# Set up the wave file to save the output:
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)

client = genai.Client(api_key=google_api_key)

response = client.models.generate_content(
   model="gemini-2.5-flash-preview-tts",
   contents=story_teller(json_data['story_context']),
   config=types.GenerateContentConfig(
      response_modalities=["AUDIO"],
      speech_config=types.SpeechConfig(
         voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
               voice_name='Kore',
            )
         )
      ),
   )
)

data = response.candidates[0].content.parts[0].inline_data.data

file_name='out.wav'
wave_file(file_name, data) # Saves the file to current directory