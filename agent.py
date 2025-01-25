from openai import OpenAI
import json
from tools import tools, handle_tool_call

CONFIG = json.load(open('config.json'))
API_KEY = CONFIG["api_key"]
MODEL = CONFIG["model"]

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

def send_messages(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
    return response.choices[0].message

system_prompt = {
    "role": "system",
    "content": """
    You are an assistant that uses external tools. When receiving requests:
    1. Analyze if specific tools are needed
    2. Call tools only when necessary
    3. ALWAYS use final_response tool to deliver the final answer
    
    Usage examples:
    - Weather request: get_weather → final_response("Rome has 24℃")
    - Task scheduling: schedule_task → final_response("Reminder set for [datetime]")
    - Math calculations: calculator → final_response("Result is 42")
    
    Guidelines:
    1. Keep conversations concise
    2. No extra comments after final_response
    3. Always format final response for end-user
    """
}

def process_conversation(messages):
    final_response = None
    iteration = 1
    
    while True:
        print(f"\n=== Conversation iteration {iteration} ===")
        response = send_messages(messages)
        
        if response.content:
            print(f"Assistant initial response: {response.content}")
        
        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": response.tool_calls
        })
        
        if not response.tool_calls:
            final_response = response.content
            break
            
        for tool_call in response.tool_calls:
            if tool_call.function.name == "final_response":
                final_args = json.loads(tool_call.function.arguments)
                final_response = final_args['content']
                print(f"\n=== Final Response ===")
                break
                
            tool_result = handle_tool_call(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": tool_result
            })
            print(f"Tool result: {tool_result}")
        
        if final_response:
            break
            
        iteration += 1
            
    return final_response

# Example usage
messages = [
    system_prompt,
    {"role": "user", "content": "Remind me to do the report tomorrow at 15:30"}
]

print("\n=== Starting Conversation ===")
print(f"User: {messages[1]['content']}")
result = process_conversation(messages)
print(f"\n=== Final Answer ===")
print(result)
