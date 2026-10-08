import os
import operator
import json
from typing import Annotated, List, Optional, TypedDict, Dict, Literal, Any
import httpx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from langgraph.checkpoint.memory import MemorySaver
from groqDay3_schema_enforcement import(
    UserQueryClassification,
    EnterpriseQueryClassifier,
    IntentCategory,
    ExtractedEntity
)

load_dotenv()

class AgentState(TypedDict):
    user_query: str
    classification: Optional[UserQueryClassification]
    tool_output: Optional[Dict[str, Any]]
    final_response : Optional[str]
    messages : Annotated[List[str], operator.add]

class ChatRequest(BaseModel):
    query: str
    thread_id: str

class ChatResponse(BaseModel):
    final_text: str
    tool_used: str
    audit_logs: List[str]
    
def find_target_entity(entities: List[ExtractedEntity], fallback :str ="unsupported resource")->str:
    """Helper: Selects the highest-confidence entity extracted during Day 3 triage."""
    if not entities:
        return fallback
    entity = sorted(entities, key=lambda e: e.confidence, reverse = True)
    return entity[0].entity_name

def run_cluster_diagnostics(cluster_id: str)->Dict[str, Any]:
    """Tool 1: Simulates hitting the MongoDB Atlas Telemetry API."""
    mock_telemetry_db = {
        "prod-db-01":{
            "status":"degraded",
            "acitve_connections":"4980/5000 (99.6% saturation)",
            "primart_error":"HTTP 504 gateway Timeout on replica-set-02",
            "automated_miitgation":"Connection pool auto-scaled to 10,000 state locks purged"
        }
    }
    return mock_telemetry_db.get(
                cluster_id.lower(),{
                    "cluster_id": cluster_id,
                    "status":"investigation",
                    "active_connections":"Normal(42%)",
                    "automated_mitigation":"diagnostics ping computed, no packet loss detected"
                }
            )
        

def fetch_billing_ledger(query_summary: str)->Dict[str, Any]:
    """Tool 2: Simulates querying the Enterprise Stripe/SAP Billing API."""
    return {
        "account_tier":"Enterprise Annual plan",
        "current_payment_method":"Corporate Visa Ending in 4421 (Expiring 10/2026)",
        "self_serve_portal_url":"https://coreai/internal/billing/payment_methods",
        "required_role":"FINANCE_ADMIN",
        "ledger_note":f"verified active subscriptions for {query_summary}"
    }
def qurantine_api_endpoint(endpoint_path: str)->Dict[str, Any]:
    """Tool 3: Simulates triggering an automated Kong/Cloudflare WAF quarantine."""
    return{
        "target_endpoint":endpoint_path,
        "containment_status":"QURANTINED_AND_RATE_LIMITED",
        "revoked_tokens":44,
        "soc_incident_id":"SOC-2026-9941",
        "pending_action":"Awaiting Mandatory Human SOC approval before restoring traffic"
    }
    
#updating previous days code to new since we have added a new node at the end of every node that is resolution synthesizer

def triage_node(state: AgentState)->dict:
    """Node 1: Calls Groq via Instructor to classify the query and extract entities."""
    print("\n-----[Node : Triage] calling groq classifier -------")
    classifier = EnterpriseQueryClassifier()
    result = classifier.classify_with_instructor(state["user_query"])
    log_entry =(f"[Triage Node] Intent = {result.primary_intent.value} |"
                f"Urgency  = {result.urgency_level.value} |"
                f"Human Review = {result.requires_human_review}")
    return {
        "classification":result,
        "messages":[log_entry]
    }
    
def technical_support_node(state: AgentState)->dict:
    """Node 2A: Passes extracted cluster entity into run_cluster_diagnostics()."""
    print("--- [NODE: Tech Support] Executing run_cluster_diagnostics() tool... ---")    
    cls_data = state["classification"]
    
    # Filter out pure numbers like '504' so we pass the cluster name ('prod-db-01') to the tool
    non_numeric = [ e for e in cls_data.entities if not e.entity_name.isdigit()]
    target_cluster= find_target_entity(non_numeric or cls_data.entities , fallback="prod-db-01")
    telemetry = run_cluster_diagnostics(target_cluster)
    return{
        "tool_output":{"tool_used":"run_cluster_diagnostics", "target":target_cluster, "data":telemetry},
        "messages":[f"[Tech Support Tool] Executed diagnostics on {target_cluster}->status: {telemetry.get('status')}"]
    }

def billing_node(state: AgentState)->dict:
    """Node 2B: Executes fetch_billing_ledger() tool."""
    print("--- [NODE: Billing] Executing fetch_billing_ledger() tool... ---")
    cls_data = state["classification"]
    ledger_data = fetch_billing_ledger(cls_data.summary)
    return{
        "tool_output":{"tool_used":"fetch_billing_ledger", "data":ledger_data},
        "messages":[f"[Billing Tool] Retrieval ledge for tier :{ledger_data['account_tier']}" ]
    }

def security_soc_node(state: AgentState)->dict:
    """Node 2C: Passes extracted endpoint entity into quarantine_api_endpoint()."""
    print("--- [NODE: Security SOC] Executing quarantine_api_endpoint() tool... ---")
    cls_data = state["classification"]
    target_endpoint = find_target_entity(cls_data.entities, fallback="/v2/reporting")
    containment_report = qurantine_api_endpoint(target_endpoint)
    return{
        "tool_output":{"tool_used":"qurantine_api_endpoint", "target":"target_endpoint", "data":containment_report},
        "messages": [
            f"[Security Tool] Quarantined '{target_endpoint}' "
            f"(Incident ID: {containment_report['soc_incident_id']})"
        ]
    }
    
def general_support_node(state: AgentState)->dict:
    """Node 2D: Fallback queue tool."""
    print("--- [NODE: General Support] Creating standard support ticket... ---")
    return{
        "tool_output": {
            "tool_used": "create_standard_ticket",
            "telemetry": {"queue": "Tier-1", "sla_hours": 24},
        },
        "messages": ["[General Tool] Opened standard Tier-1 ticket (24h SLA)."],
    }


def resolution_synthesizer_node(state: AgentState)->dict:
    """Node 3: Synthesizes the final user response grounded in state['tool_output']."""
    print("--- [NODE: Synthesizer] Generating final response from tool output... ---")
    cls_data = state["classification"]
    tool_data = state["tool_output"]
    
    raw_client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        http_client=httpx.Client(verify=False)
    )
    completions  = raw_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an enterprise operations copilot. "
                    "Write a concise, professional 3-sentence resolution update for the user "
                    "DO NOT use Markdown formatting, bolding, or asterisks in your response."
                    "based STRICTLY on the Classification Metadata and Executed Tool Telemetry. "
                    "Cite exact numbers, URLs, or Incident IDs from the telemetry. "
                    "If HumanReview is True, explicitly state that a human reviewer has been paged."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User Query: {state['user_query']}\n\n"
                    f"Classification Metadata: Intent={cls_data.primary_intent.value}, "
                    f"Urgency={cls_data.urgency_level.value}, "
                    f"HumanReview={cls_data.requires_human_review}\n\n"
                    f"Executed Tool Telemetry:\n{json.dumps(tool_data, indent=2)}"
                ),
            }
        ]
    )
    final_text = completions.choices[0].message.content.strip()
    return{
        "final_response": final_text,
        "messages":[
            "[Synthesizer] Final grounded Response generated."
        ]
    }

def route_query(
    state: AgentState,
) -> Literal["tech_support", "billing", "security_soc", "general_support"]:
    """Inspects the validated IntentCategory Enum with zero LLM calls."""
    intent = state["classification"].primary_intent
    print(f"--- [ROUTER] Traffic Cop routing intent: {intent.value} ---")

    if intent == IntentCategory.SECURITY_ALERT:
        return "security_soc"
    elif intent in (IntentCategory.TECHNICAL_SUPPORT, IntentCategory.DATA_PIPELINE):
        return "tech_support"
    elif intent == IntentCategory.BILLING_ISSUE:
        return "billing"
    else:
        return "general_support"
    
workflow = StateGraph(AgentState)

workflow.add_node("triage",triage_node)
workflow.add_node("tech_support", technical_support_node)
workflow.add_node("billing", billing_node)
workflow.add_node("security_soc", security_soc_node)
workflow.add_node("general_support", general_support_node)
workflow.add_node("synthesizer", resolution_synthesizer_node)

#entry edge
workflow.add_edge(START, "triage")
workflow.add_conditional_edges("triage", route_query,{
    "tech_support":"tech_support",
    "billing":"billing",
    "security_soc":"security_soc",
    "general_support":"general_support"
})

# All department nodes converge into the Synthesizer node
workflow.add_edge("tech_support", "synthesizer")
workflow.add_edge("billing", "synthesizer")
workflow.add_edge("security_soc", "synthesizer")
workflow.add_edge("general_support", "synthesizer")

# Synthesizer exits to END
workflow.add_edge("synthesizer", END)

memory = MemorySaver()
agent_app = workflow.compile(checkpointer=memory)

app = FastAPI(title="CoreAi Triage API", version="1.0.0")

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        config = {"configurable":{"thread_id": request.thread_id}}
        final_state = agent_app.invoke(
            {"user_query":request.query},
            config = config
        )
        return ChatResponse(
            final_text = final_state["final_response"],
            tool_used = final_state.get("tool_output", {}).get("tool_used", "none"),
            audit_logs = final_state["messages"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    print("\n🚀 Starting Epsilon COREai REST Server...")
    print("Test via cURL or Postman at: POST http://localhost:8000/api/v1/chat")
    uvicorn.run(app, host="0.0.0.0", port=8000)