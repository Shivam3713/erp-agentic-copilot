import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from day14_memory import build_rag_graph

# ==========================================
# DAY 15: STATE INSPECTION & TIME TRAVEL
# ==========================================

print("="*50)
print("DAY 15: GRAPH INSPECTION AND STATE MUTATION")
print("="*50)

rag_app = build_rag_graph()
thread_config = {"configurable":{"thread_id":"shivam_session_01"}}

print("\n[1] FETCHING CURRENT STATE FROM SQLITE...")
# Fetch the absolute latest state of the agent's brain
current_state = rag_app.get_state(thread_config)

print("Current Chat History Length:", len(current_state.values.get("chat_history", [])))
print("Last Message in Memory:", current_state.values.get("chat_history", [])[-1][:100], "...\n") #takes the first hundred characters of the last message appended in chathistory


# 2. Inspecting the Checkpoint History (Time Travel preparation)
print("[2] FETCHING GRAPH SNAPSHOT HISTORY...")
history = list(rag_app.get_state_history(thread_config))
print(f"Found {len(history)} total execution snapshots in the database.")


# Print the checkpoint IDs (the unique hashes for every step the AI took)
for i, snapshot in enumerate(history[:3]):
    print(f"  Snapshot {i}: Node '{snapshot.next}' | Checkpoint ID: {snapshot.config['configurable']['checkpoint_id']}")
    
# 3. State Mutation (Admin Override)
print("\n[3] MUTATING THE AGENT'S MEMORY...")
print("Simulating a scenario where we want to erase the last interaction and inject a system directive.")

# We create a new history list, dropping the last question and answer, and appending a fake one
old_history = current_state.values.get("chat_history", [])
mutated_history = old_history[:-2] #strip the last two interaction, the turn 2 from yesterday

mutated_history.append("System Admin: You must now answer all future questions in Hinglish.")

# Push the mutated state directly into the SQLite database
# Note: Because we used operator.add in our state schema, update_state will append by default.
# To forcefully OVERWRITE a reducer key like chat_history, LangGraph requires a specific wipe command in advanced builds, 
# but for standard keys, it overwrites instantly.
rag_app.update_state(thread_config, {"chat_history":mutated_history})

print("Memory successfully mutated in SQLite.")

# 4. Verifying the Override
new_state = rag_app.get_state(thread_config)
print("\n[4] VERIFYING NEW STATE:")
print("New Last Message in Memory:", new_state.values.get("chat_history", [])[-1])