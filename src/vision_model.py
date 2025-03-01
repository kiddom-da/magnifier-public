import os
import base64
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
# from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class MagnifierItem(BaseModel):
    cycle_id: int
    page_number: int | str | None
    text_after_symbol: str = Field(description="The text that follows the magnifier symbol")

class MagnifierPage(BaseModel):
    magnifier_items: List[MagnifierItem]

class VisionProcessor:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.gemini_client = genai.GenerativeModel('gemini-2.0-flash')
        self.qwen_client = OpenAI(api_key=os.getenv('QWEN_API_KEY'),
                                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",)
        
        # self.qwen2_client = genai.GenerativeModel('qwen2.5-VL-70b')
    def detect_magnifier_gemini(self, image_path: str) -> bool:
        """
        Uses Gemini API to detect magnifier symbol in an image.
        """
        try:
            # Load and encode the image
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            
            # Convert to base64 for API
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create a Gemini model instance
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Prepare the image for the model
            image_part = {
                "mime_type": "image/png",
                "data": encoded_image
            }
            
            # Create a more specific prompt that requests JSON output
            prompt = f"""
            Is there a magnifier symbol in this page's margins? answer 'yes' or 'no'
            """
            
            # Generate content with the image and prompt
            response = model.generate_content([prompt, image_part])
            
            # Clean up response - get only true/false
            result = response.text.strip().lower()
            
            # Ensure we only get true or false
            is_found = result == 'yes'
            
            return is_found
            
        except Exception as e:
            logging.error(f"Error in symbol detection: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def detect_magnifier_qwen(self, image_path: str) -> bool:
        """
        Uses qwen2.5-VL-70b API to detect magnifier symbol in an image.
        """
        try:
            # Load and encode the image
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')

            # Create a O1 model instance
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """are there any magnifier glasses symbols in this page's margin? answer 'yes' or 'no'"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }
            ]

            response = self.qwen_client.chat.completions.create(
                    model="qwen2.5-vl-72b-instruct",
                    messages=messages,
                    temperature=0.0
            )
            result = response.choices[0].message.content.strip().lower()
            print('qwen', result)
            return result == 'yes'

        except Exception as e:
            logging.error(f"Error in detect_magnifier_o1: {str(e)}")
            return False

    def detect_magnifier_gpt(self, image_path: str) -> bool:
        """
        Uses o1 API to detect magnifier symbol in an image.
        """
        try:
            # Load and encode the image
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')

            messages = [
                {
                    "role": "user",
                    "content": [

                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        },
                        {
                            "type": "text",
                            "text": """are there any magnifier symbols in this page's margins? answer 'yes' or 'no'"""
                        },
                    ]
                }
            ]

            response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    response_format={"type": "text"},
                    temperature=1,
                    max_completion_tokens=1,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
            )
            result = response.choices[0].message.content.strip().lower()
            print(result)
            return result == 'yes'

        except Exception as e:
            logging.error(f"Error in detect_magnifier_o1: {str(e)}")
            return False

    def extract_text(self, image_path: str) -> dict:
        """
        Extract text and metadata from the page with magnifier using GPT-4 Vision
        """
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the magnifier item from the page image, item by item. if there's no magnifier item, return an empty list."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        }
                    ]
                }
            ]

            try: 
                response = self.openai_client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.0,
                    response_format=MagnifierPage
                )

                return response.choices[0].message.parsed
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                return None
            
        except Exception as e:
            logging.error(f"Error in extract_text: {str(e)}")
            return {"page_number": None, "text": None} 
        






