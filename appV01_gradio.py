import gradio as gr
from dotenv import load_dotenv
import os
import json
import wave
import tempfile
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing required libraries with error handling
try:
    from mem0 import MemoryClient, Memory
    MEM0_AVAILABLE = True
except ImportError:
    logger.warning("mem0 library not available. Memory features will be disabled.")
    MEM0_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    logger.error("Google AI library not available. Please install google-generativeai")
    GOOGLE_AI_AVAILABLE = False

class StorytellingApp:
    def __init__(self):
        self.setup_environment()
        self.initialize_clients()
    
    def setup_environment(self):
        """Load environment variables"""
        try:
            load_dotenv()
            self.google_api_key = os.getenv('GOOGLE_API_KEY')
            self.mem_api_key = os.getenv('MEM_API_KEY')
            
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
                
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            raise
    
    def initialize_clients(self):
        """Initialize API clients"""
        try:
            if not GOOGLE_AI_AVAILABLE:
                raise ImportError("Google AI library not available")
                
            self.google_client = genai.Client(api_key=self.google_api_key)
            
            # Initialize memory client if available
            if MEM0_AVAILABLE and self.mem_api_key:
                self.mem_client = MemoryClient(api_key=self.mem_api_key)
            else:
                self.mem_client = None
                logger.warning("Memory client not initialized")
                
        except Exception as e:
            logger.error(f"Client initialization failed: {e}")
            raise
    
    def google_image_desc_context(self, file_path: str) -> dict:
        """Extract story context and description from image"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file not found: {file_path}")
            
            sys_prompt = """You are a very talented storyteller who can create amazing, attention-grabbing stories just by looking at pictures. 
            Your main task as a storyteller is to give the best context for a story so that another model can generate seamlessly.
            
            Your main 2 tasks are to give:
            1. Complete description of the image
            2. Give the base context as well as ending context for the story
            
            You will always return the response in JSON format:
            {
                "story_context": "here we will have the context for the story",
                "image_description": "here we will have the complete description of the image"
            }"""
            
            # Upload file to Google AI
            my_file = self.google_client.files.upload(file=file_path)
            
            # Generate content
            response = self.google_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[my_file, sys_prompt],
                config={
                    "response_mime_type": "application/json",
                },
            )
            
            result = json.loads(response.text)
            
            # Validate response structure
            if 'story_context' not in result or 'image_description' not in result:
                raise ValueError("Invalid response format from AI model")
                
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise ValueError("Failed to parse AI response")
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise
    
    def story_teller(self, context: str) -> str:
        """Generate story from context"""
        try:
            if not context or not context.strip():
                raise ValueError("Context cannot be empty")
            
            prompt = f"""You are a very good storyteller who can create attention-grabbing stories just with the use of context. 
            Your job is to create a story based on the provided context: {context}
            Create a story within 150 words that is engaging and complete."""
            
            response = self.google_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            
            if not response.text:
                raise ValueError("Empty response from story generation")
                
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Story generation failed: {e}")
            raise
    
    def create_audio_story(self, story_text: str) -> str:
        """Convert story text to audio"""
        try:
            if not story_text or not story_text.strip():
                raise ValueError("Story text cannot be empty")
            
            response = self.google_client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=story_text,
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
            
            if not response.candidates or not response.candidates[0].content.parts:
                raise ValueError("No audio data received from TTS service")
            
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            
            # Create temporary file for audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            self.save_wave_file(temp_file.name, audio_data)
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise
    
    def save_wave_file(self, filename: str, pcm_data: bytes, channels: int = 1, 
                      rate: int = 24000, sample_width: int = 2):
        """Save PCM data as WAV file"""
        try:
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                wf.writeframes(pcm_data)
        except Exception as e:
            logger.error(f"Wave file creation failed: {e}")
            raise
    
    def process_image_to_story(self, image_file) -> Tuple[str, str, str, Optional[str]]:
        """Main processing function for Gradio interface"""
        try:
            if image_file is None:
                return "❌ Error: Please upload an image", "", "", None
            
            # Analyze image
            try:
                analysis_result = self.google_image_desc_context(image_file)
                image_description = analysis_result['image_description']
                story_context = analysis_result['story_context']
            except Exception as e:
                return f"❌ Image analysis failed: {str(e)}", "", "", None
            
            # Generate story
            try:
                story_text = self.story_teller(story_context)
            except Exception as e:
                return f"❌ Story generation failed: {str(e)}", image_description, story_context, None
            
            # Generate audio
            try:
                audio_file = self.create_audio_story(story_text)
                return "✅ Story generated successfully!", image_description, story_context, story_text, audio_file
            except Exception as e:
                logger.warning(f"Audio generation failed: {e}")
                return "⚠️ Story generated but audio creation failed", image_description, story_context, story_text, None
                
        except Exception as e:
            logger.error(f"Unexpected error in processing: {e}")
            return f"❌ Unexpected error: {str(e)}", "", "", None

def create_gradio_interface():
    """Create and configure Gradio interface"""
    try:
        app = StorytellingApp()
    except Exception as e:
        logger.error(f"Failed to initialize app: {e}")
        
        # Create a fallback interface that shows the error
        def error_interface(image):
            return f"❌ Application initialization failed: {str(e)}", "", "", None
        
        interface = gr.Interface(
            fn=error_interface,
            inputs=gr.Image(type="filepath", label="Upload Image"),
            outputs=[
                gr.Textbox(label="Status", lines=2),
                gr.Textbox(label="Image Description", lines=3),
                gr.Textbox(label="Story Context", lines=3),
                gr.Audio(label="Story Audio")
            ],
            title="🎭 AI Storyteller - Error",
            description="Application failed to initialize. Please check your API keys and dependencies.",
        )
        return interface
    
    # Create the main interface
    def process_wrapper(image):
        try:
            return app.process_image_to_story(image)
        except Exception as e:
            logger.error(f"Processing wrapper error: {e}")
            return f"❌ Processing failed: {str(e)}", "", "", None
    
    interface = gr.Interface(
        fn=process_wrapper,
        inputs=[
            gr.Image(
                type="filepath", 
                label="📸 Upload Image",
                height=300
            )
        ],
        outputs=[
            gr.Textbox(
                label="📋 Status", 
                lines=2,
                show_copy_button=True
            ),
            gr.Textbox(
                label="🖼️ Image Description", 
                lines=4,
                show_copy_button=True
            ),
            gr.Textbox(
                label="📖 Story Context", 
                lines=4,
                show_copy_button=True
            ),
            gr.Textbox(
                label="📚 Generated Story", 
                lines=6,
                show_copy_button=True
            ),
            gr.Audio(
                label="🎵 Story Audio",
                type="filepath"
            )
        ],
        title="🎭 AI Storyteller",
        description="""
        Upload an image and let AI create an amazing story for you! 
        
        **Features:**
        - 🔍 Analyzes your image to understand the scene
        - 📝 Creates engaging story context
        - ✍️ Generates a complete story (150 words)
        - 🎤 Converts story to speech audio
        
        **Requirements:**
        - Valid GOOGLE_API_KEY in your .env file
        - Supported image formats: JPG, PNG, GIF, BMP
        """,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 100% !important;
            width: 100% !important;
        }
        .contain {
            max-width: 100% !important;
        }
        """,
        examples=[
            # You can add example images here if you have them
        ],
        cache_examples=False,
        allow_flagging="never"
    )
    
    return interface

if __name__ == "__main__":
    try:
        # Check if required environment variables exist
        load_dotenv()
        if not os.getenv('GOOGLE_API_KEY'):
            print("❌ Error: GOOGLE_API_KEY not found in environment variables")
            print("Please create a .env file with your Google AI API key")
            exit(1)
        
        # Launch the interface
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,       # Default Gradio port
            share=False,            # Set to True to create public link
            debug=True              # Enable debug mode
        )
        
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        print(f"❌ Application launch failed: {e}")