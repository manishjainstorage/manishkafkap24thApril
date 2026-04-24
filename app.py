import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Set page configuration
st.set_page_config(page_title="HuggingFace Text App", layout="wide")
st.title("🤗 Text to Text & Text to Image Generator")

# Sidebar for model selection and caching
st.sidebar.header("Settings")

# Radio button for selecting mode
mode = st.radio(
    "Select Mode:",
    ["Text to Text (Summarization)", "Text to Image (Image Generation)"],
    help="Choose between text summarization or image generation"
)

# Main input textbox
user_input = st.text_area(
    "Enter your text:",
    placeholder="Type something here...",
    height=150,
    help="Provide input text for processing"
)

# Process based on selected mode
if user_input:
    try:
        if mode == "Text to Text (Summarization)":
            st.subheader("📝 Text Summarization")
            
            with st.spinner("Loading summarization model..."):
                # Load summarization pipeline
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            
            with st.spinner("Generating summary..."):
                # Ensure input is long enough for summarization
                if len(user_input.split()) > 50:
                    summary = summarizer(user_input, max_length=150, min_length=50, do_sample=False)
                    summary_text = summary[0]['summary_text']
                    
                    st.success("✅ Summary Generated!")
                    st.write("**Original Text:**")
                    st.info(user_input)
                    st.write("**Summarized Text:**")
                    st.success(summary_text)
                else:
                    st.warning("⚠️ Text is too short for summarization. Please provide at least 50 words.")
        
        else:  # Text to Image
            st.subheader("🎨 Text to Image Generation")
            
            with st.spinner("Loading image generation model... (This may take a moment)"):
                # Check if CUDA is available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                st.info(f"Using device: {device.upper()}")
                
                # Load Stable Diffusion pipeline
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float32
                )
                pipe = pipe.to(device)
            
            with st.spinner("Generating image... (This may take 30-60 seconds)"):
                # Generate image
                image = pipe(user_input, num_inference_steps=50).images[0]
                
                st.success("✅ Image Generated!")
                st.write("**Prompt:**")
                st.info(user_input)
                st.write("**Generated Image:**")
                st.image(image, use_column_width=True)
                
                # Add download button for the image
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                st.download_button(
                    label="Download Image",
                    data=img_byte_arr,
                    file_name="generated_image.png",
                    mime="image/png"
                )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Make sure you have the required libraries installed: `pip install streamlit transformers diffusers torch pillow`")

else:
    st.info("👈 Enter text in the textbox above to get started!")

# Footer
st.markdown("---")
st.markdown("""
**Note:** 
- Text to Image generation requires significant computing power and may take time
- Using GPU (CUDA) will be much faster than CPU
- Install required packages: `pip install streamlit transformers diffusers torch pillow`
""")
