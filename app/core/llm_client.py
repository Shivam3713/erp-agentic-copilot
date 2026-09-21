import os
from google import genai
from google.genai import types
import anthropic
from dotenv import load_dotenv
load_dotenv()


class PromptTemplate:
    @staticmethod
    def format(template_str: str, **kwargs)->str:
        try:
            return template_str.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"incorrect arguments : {e}")


class LLM_client:
    
    def __init__(self, provider:str="Gemini"):
        
        self.provider = provider.lower()
        if self.provider == "gemini":
            self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) # os.getenv fetches your secure key; genai.Client logs into Google.
            self.gemini_model = "gemini-3.6-flash"
            
        elif self.provider == "anthropic":
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.anthropic_model = "claude-3.5-sonnet-20241022"
        
        else:
            raise ValueError(f"unsupported choose gemini or anthropic")
    
    def generate(self, system_prompt:str, user_prompt: str)->str:
        if self.provider == "gemini":
            response = self.gemini_client.models.generate_content(
                model = self.gemini_model,
                contents=user_prompt,
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            )
            return response.text
        
        elif self.provider == "anthropic":
            response = self.anthropic_client.messages.create(
                model = self.anthropic_model,
                max_tokens = 1024,
                system = system_prompt,
                messages = [{
                    "role":"user","content":"user_prompt"
                }]
            )
            return response.content[0].text
        