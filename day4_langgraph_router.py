import operator
from typing import TypedDict, Optional, Annotated, List, Literal
from langgraph.graph import StateGraph, START, END

from groqDay3_schema_enforcement import (EnterpriseQueryClassifier, UrgencyLevel, UserQueryClassification, IntentCategory)
from dotenv import load_dotenv
load_dotenv()

#define the clipboard for the route query to check and reroute the request during the graph callback

class AgentState(TypedDict):
    user_query:str
    classification: Optional[UserQueryClassification]
    messages: Annotated[List[str], operator.add]


#entry worker or node into the graph from start
def triage_node(state: AgentState)->dict:
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
    
#step 4 would be to write the downstream nodes

def technical_support_node(state: AgentState)->dict:
    print("-------[Node: Tech Support] Running Automated Diagnostics --------")
    cls_data = state["classification"]
    extracted = [(e.entity_name, e.entity_type) for e in cls_data.entities]
    return{
        "messages": [f"[Tech Support] query diagnostics for entities {cls_data.entities}"]
    }
    
def security_escalation_node(state: AgentState)->dict:
    print("-------[Node: Security SOC] paging on call security engineer -------")
    cls_data = state["classification"]
    return {
        "messages": [
            f"[Security SOC] Paged human reviewer={cls_data.requires_human_review}. "
            f"Summary: {cls_data.summary}"
        ]
    }
def billing_node(state: AgentState)->dict:
    print("------[Node: Billing] Fteching account ledgeer-------")
    cls_data = state["classification"]
    return {
        "messages":[f"[Billing] Opened Billing Tickek"
                    f"[Summary]: {cls_data.summary}"]
    }

def general_support_node(state: AgentState)->dict:
    print(f"-----[Node: General Support] routing for standard support desk ------")
    cls_data = state["classification"]
    return{
        "messages":"f[General Support] Assgined to standard triage node"
    }

#now our router that will conditionally route the request based on our condition

def route_query(state: AgentState)->Literal["tech_support", "billing","security_soc", "general_support"]:
    cls_data = state["classification"]
    intent = cls_data.primary_intent
    print(f"----[ROUTER] Evaluating Intent ENUM :{intent.value}")
    if intent == IntentCategory.SECURITY_ALERT:
        return "security_soc"
    elif intent in (IntentCategory.TECHNICAL_SUPPORT, IntentCategory.DATA_PIPELINE):
        return "tech_support"
    elif intent == IntentCategory.BILLING_ISSUE: return "billing"
    else: return "general_support"


#now to write our stategraph connecting the nodes, edges
workflow = StateGraph(AgentState)

#register every worker ndoe
workflow.add_node("triage", triage_node)
workflow.add_node("tech_support", technical_support_node)
workflow.add_node("billing", billing_node)
workflow.add_node("security_soc", security_escalation_node)
workflow.add_node("general_support", general_support_node)

#connect start with triage
workflow.add_edge(START, "triage")

#connect triage with router query with conditional
workflow.add_conditional_edges("triage", route_query, {
    "tech_support":"tech_support",
    "billing":"billing",
    "security_soc":"security_soc",
    "general_support":"general_support"
})

#connect all worker node to end
workflow.add_edge("tech_support", END)
workflow.add_edge("billing", END)
workflow.add_edge("security_soc", END)
workflow.add_edge("general_support", END)

app = workflow.compile()
#write the main function to run and test

if __name__ == "__main__":
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
    for idx, query in enumerate(test_queries, 1):
        print(f"Graph Run -----{idx}----")
        initial_state : AgentState = {
            "user_query":query,
            "classification":None,
            "messages":[f"[System] Received_query :{query}"]
        }
        final_state = app.invoke(initial_state)
        print(f"Final Audit [State Message] ------")
        for msg in final_state["messages"]:
            print(f"-> {msg}")
    