import gradio as gr
import google.generativeai as genai
from huggingface_hub import InferenceClient
from PIL import Image
import io
from dotenv import load_dotenv
import os

# Initialize Gemini
# Load environment variables
load_dotenv()

# Initialize Gemini
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Inference Client
client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
)

def process_and_generate(image):
    try:
        # Step 1: Analyze the image
        prompt = """
        Analyze this image in great detail. Please provide:
        1. A comprehensive description of all visible elements
        2. Information about people, their appearance, expressions, and activities
        3. Details about the environment, setting, and background
        4. Analysis of colors, lighting, mood, and composition
        5. Any notable objects and their significance
        6. Overall context and meaning if apparent
        
        Be thorough and descriptive, providing at least 150 words of analysis.
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        generation_config = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_data = image_bytes.getvalue()
        
        content_parts = [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_data}}
        ]
        
        response = model.generate_content(
            content_parts,
            generation_config=generation_config,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]
        )
        
        analysis_text = response.text
        
        # Step 2: Generate image from the analysis text
        image = client.text_to_image(
            analysis_text,
            model="black-forest-labs/FLUX.1-dev",
        )
        
        return analysis_text, image
        
    except Exception as e:
        return f"Error processing image: {str(e)}", None

with gr.Blocks(title="Image Analysis and Regeneration") as demo:
    gr.Markdown("### Upload an Image to Analyze and Regenerate")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        submit_btn = gr.Button("Analyze and Generate")
    with gr.Row():
        text_output = gr.Textbox(label="Gemini Analysis", lines=15)
        image_output = gr.Image(label="Generated Image")
    
    submit_btn.click(
        fn=process_and_generate,
        inputs=image_input,
        outputs=[text_output, image_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)