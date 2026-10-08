import os
import instructor
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


#step 2 make enum classes 

class IntentCategory(str, Enum):
    ACCOUNT_ENQUIRY="account_enquiry"
    TECHNICAL_SUPPORT="technical_support"
    BILLING_ISSUE="billing_issue" 
    DATA_PIPELINE="data_pipeline" 
    FEATURE_REQUEST="feature_request" 
    SECURITY_ALERT="security_alert" 
    UNKNOWN="unknown"
    
class UrgencyLevel(str, Enum):
    LOW="low"
    MEDIUM="medium"
    HIGH="high"
    CRITICAL="critical"

#step 3 make the entity to be extracter
class ExtractedEntity(BaseModel):
    entity_name :str = Field(..., description="name or indentifier of the entity")
    entity_type :str = Field(..., description="type category example cluster id, api endpoint, user id")
    confidence  :float = Field(..., ge=0.0, le=1.0, description="confidence scores between 0.0 and 1.0")
    
#jo output humein chahiye uska schema likhna hai aur uske validation jo humne khud define kia hai
class UserQueryClassification(BaseModel):
    """production schema for classifying user query and enterprise customer queries"""
    primary_intent:IntentCategory = Field(..., description="the main intent of the query")
    secondary_intent:List[IntentCategory] = Field(default_factory=list, description="ancilliary intents detected")
    confidence_score : float = Field(..., ge=0.0, le=1.0, description="overall classfication confidence score")
    urgency_level :UrgencyLevel = Field(default=UrgencyLevel.LOW, description="assessed business urgency" )
    summary :str = Field(..., min_length=10, max_length=256, description="concise executive summary of query")
    entities:List[ExtractedEntity] = Field(default_factory=list, description="entities passed from the query")
    requires_human_review : bool = Field(default=False, description="flag if human review is mandatory")
    
    @field_validator("summary")
    @classmethod
    def validateSummary(cls, v:str)->str:
        cleaned = v.strip()
        if "as an ai" in cleaned.lower() or "as an llm" in cleaned.lower():
            raise ValueError("summary must not contain generic boilerplate")
        return cleaned
    
    @model_validator(mode="after")
    def enforce_enterprise_business_rules(self)->"UserQueryClassification":
        if self.urgency_level == UrgencyLevel.CRITICAL and not self.requires_human_review:
            raise ValueError("all ciritical urgency require human review which must be set True")
        if self.primary_intent == IntentCategory.SECURITY_ALERT:
            self.urgency_level = UrgencyLevel.CRITICAL
            self.requires_human_review= True
        return self
    


#ab calling google genai using instructor to do the needful
class EnterpriseQueryClassifier:
    def __init__(self, api_key: Optional[str]=None, model:str = "gemini-3.6-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.raw_client = genai.Client(api_key=self.api_key)
        self.client = instructor.from_genai(self.raw_client)
        self.model = model
    
    def classify_with_instructor(self, query:str, max_retries:int = 3):
        """extract structured classification with automated instructor asking for validation failure"""
        return self.client.chat.completions.create(
            model = self.model,
            response_model=UserQueryClassification,
            max_retries=max_retries,
            messages=[{
                "role":"system",
                "content":(
                    "you are a senior triage for capgemini. "
                    "analyze enterprise user prompts and output strictly validated schemas"
                )
            },{
                "role":"user",
                "content":f"classify the following user {query}\n\n"
            }]
        )
    
    #this is for writing if we did not user instructor
    def classify_native_genai_response(self, query:str)->UserQueryClassification:
        response = self.raw_client.models.generate_content(
            model = self.model,
            contents=f"classify the following user {query}\n\n",
            config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=UserQueryClassification,
                            system_instruction="you are an enterpruse query classfier for capgeminin coreAI",
                        )
        )
        return UserQueryClassification.model_validate_json(response.text)
    
    
#main funtion to execute our code

if __name__ == "__main__":
    classifier = EnterpriseQueryClassifier()
    test_queries=["urgent, mongodb atlas cluster prod-db-01 is dropping connection with 504 error accorss our analytics pipeline",
                  "can you guide me through our team billing account payment method",
                  "security incident unauthorized api access detected on API endpoint /v2/reporting"]
    
    for idx, query in enumerate(test_queries, 1):
        print(f"\n----------testing case {idx}------------")
        print(f"query is : {query}\n")
        try:
            result = classifier.classify_with_instructor(query)
            print(f"primary intent: {result.primary_intent.value}")
            print(f"urgency level: {result.urgency_level.value}")
            print(f"human review needed ? : {result.requires_human_review}")
            print(f"executive summary: {result.summary}")
            print(f"extracted entities: {[(e.entity_name, e.entity_type)for e in result.entities]}")
        except Exception as e:
            print(f"execution error : {e}")
        