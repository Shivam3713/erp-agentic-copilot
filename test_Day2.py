from app.core.llm_client import LLM_client, PromptTemplate

SYSTEM_TEMPLATE ="you are an enterprise ai assistant for capgemini response in a{tone} tone."
USER_TEMPLATE = "classify the following query into a category of (support, billing, sales) {query}"

system_prompt = PromptTemplate.format(SYSTEM_TEMPLATE, tone="highly professional")
user_prompt = PromptTemplate.format(USER_TEMPLATE, query ="my account is locked and i cant view my invoice")

print(f"----testing gemini-------")
gemini_client = LLM_client(provider="gemini")
gemini_response = gemini_client.generate(system_prompt, user_prompt)
print(f"response from google \n{gemini_response}\n")

print(f"-------testing anthropic--------")
try:
    
    anthropic_client = LLM_client(provider="anthropic")
    anthropic_response = anthropic_client.generate(system_prompt, user_prompt)
    print(f"anthropic response \n {anthropic_response}\n")
except Exception as e:
    print(f"anthropic api key error {e}")