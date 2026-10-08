import os
from enum import Enum
from typing import List, Optional
import httpx
import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

load_dotenv()


# ==========================================
# ENUMS
# ==========================================

class IntentCategory(str, Enum):
    ACCOUNT_ENQUIRY = "account_enquiry"
    TECHNICAL_SUPPORT = "technical_support"
    BILLING_ISSUE = "billing_issue"
    DATA_PIPELINE = "data_pipeline"
    FEATURE_REQUEST = "feature_request"
    SECURITY_ALERT = "security_alert"
    UNKNOWN = "unknown"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ==========================================
# ENTITY MODEL
# ==========================================

class ExtractedEntity(BaseModel):
    entity_name: str = Field(
        ...,
        description="name or identifier of the entity"
    )

    entity_type: str = Field(
        ...,
        description="entity type such as cluster id, endpoint, user id"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="confidence score between 0 and 1"
    )


# ==========================================
# OUTPUT SCHEMA
# ==========================================

class UserQueryClassification(BaseModel):
    """
    Production schema for enterprise query classification.
    """

    primary_intent: IntentCategory = Field(
        ...,
        description="primary user intent"
    )

    secondary_intent: List[IntentCategory] = Field(
        default_factory=list,
        description="additional intents detected"
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="overall confidence score"
    )

    urgency_level: UrgencyLevel = Field(
        default=UrgencyLevel.LOW,
        description="business urgency level"
    )

    summary: str = Field(
        ...,
        min_length=10,
        max_length=256,
        description="executive summary"
    )

    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="entities identified from prompt"
    )

    requires_human_review: bool = Field(
        default=False,
        description="whether human review is mandatory"
    )

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        cleaned = value.strip()

        if "as an ai" in cleaned.lower():
            raise ValueError(
                "summary must not contain ai boilerplate text"
            )

        if "as an llm" in cleaned.lower():
            raise ValueError(
                "summary must not contain llm boilerplate text"
            )

        return cleaned

    @model_validator(mode="after")
    def enforce_business_rules(
        self
    ) -> "UserQueryClassification":

        if (
            self.urgency_level == UrgencyLevel.CRITICAL
            and not self.requires_human_review
        ):
            raise ValueError(
                "all critical cases require human review"
            )

        if self.primary_intent == IntentCategory.SECURITY_ALERT:
            self.urgency_level = UrgencyLevel.CRITICAL
            self.requires_human_review = True

        return self


# ==========================================
# CLASSIFIER
# ==========================================

class EnterpriseQueryClassifier:

    def __init__(
        self,
        api_key: Optional[str] = None,
        # model: str = "llama-3.3-70b-versatile",
        model: str = "openai/gpt-oss-20b"
    ):  

        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        self.raw_client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
            http_client=httpx.Client(verify=False)  # <-- Add this line
        )

        self.client = instructor.from_openai(
            self.raw_client,
            mode=instructor.Mode.JSON  # <-- Add this parameter
        )

        self.model = model

    def classify_with_instructor(
        self,
        query: str,
        max_retries: int = 3
    ) -> UserQueryClassification:

        return self.client.chat.completions.create(
            model=self.model,
            response_model=UserQueryClassification,
            max_retries=max_retries,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior enterprise support triage "
                        "specialist for Capgemini. "
                        "Classify the query accurately and return "
                        "only information required by the schema."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Classify the following query:\n\n{query}"
                    ),
                },
            ],
        )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    classifier = EnterpriseQueryClassifier()

    test_queries = [
        (
            "urgent, mongodb atlas cluster "
            "prod-db-01 is dropping connections "
            "with 504 errors across our analytics pipeline"
        ),
        (
            "can you guide me through our team "
            "billing account payment method"
        ),
        (
            "security incident unauthorized api access "
            "detected on API endpoint /v2/reporting"
        ),
    ]

    for idx, query in enumerate(test_queries, start=1):

        print(f"\n{'-' * 20}")
        print(f"TEST CASE {idx}")
        print(f"{'-' * 20}")

        print(f"\nQuery:\n{query}\n")

        try:

            result = classifier.classify_with_instructor(
                query
            )

            print(
                f"Primary Intent: "
                f"{result.primary_intent.value}"
            )

            print(
                f"Urgency Level: "
                f"{result.urgency_level.value}"
            )

            print(
                f"Human Review: "
                f"{result.requires_human_review}"
            )

            print(
                f"Confidence Score: "
                f"{result.confidence_score}"
            )

            print(
                f"Summary: "
                f"{result.summary}"
            )

            print("\nEntities:")

            for entity in result.entities:
                print(
                    f"  - {entity.entity_name}"
                    f" ({entity.entity_type})"
                    f" [{entity.confidence}]"
                )

        except Exception as e:
            print(f"Execution Error: {e}")