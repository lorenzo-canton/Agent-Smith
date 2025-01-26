from openai import OpenAI
import json
from tools import tools, handle_tool_call

CONFIG = json.load(open('config.json'))
API_KEY = CONFIG["api_key"]
MODEL = CONFIG["model"]

class Agent:
    def __init__(self):
        self.client = OpenAI(
            api_key=API_KEY, 
            base_url=CONFIG["base_url"]  # Usa il base_url dal config
        )
        self.system_prompt = {
            "role": "system",
            "content": """
            You are an assistant that uses external tools. When receiving requests:
            1. Analyze if specific tools are needed
            2. Call tools only when necessary
            3. ALWAYS use final_response tool to deliver the final answer
            
            Available tools:
            - get_weather: Get weather conditions
            - schedule_task: Schedule tasks using cron expressions
            - list_all_tasks: Get markdown formatted list of all tasks
            - delete_task_by_objective: Delete tasks containing specific text
            
            For scheduling tasks:
            1. Ask user for cron expression (e.g., "0 9 * * 1-5" for weekdays at 9am)
            2. Ask for objectives to schedule
            3. Use schedule_task tool with format:
            {
                "cron_expression": "the_cron_expression",
                "objectives": ["objective1", "objective2"]
            }
            
            Guidelines:
            1. Keep conversations concise
            2. No extra comments after final_response
            3. Always format final response for end-user
            """
        }

    def send_messages(self, messages):
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )
        return response.choices[0].message

    def process_conversation(self, messages):
        final_response = None
        iteration = 1
        
        while True:
            print(f"\n=== Conversation iteration {iteration} ===")
            response = self.send_messages(messages)
            
            if response.content:
                print(f"Assistant initial response: {response.content}")
            
            # Check for final_response tool call first
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call.function.name == "final_response":
                        final_args = json.loads(tool_call.function.arguments)
                        final_response = final_args['content']
                        # Add only the final response to messages
                        messages.append({
                            "role": "assistant",
                            "content": final_response
                        })
                        return final_response
            
            # Store the assistant's message
            assistant_message = {
                "role": "assistant",
                "content": response.content
            }
            
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response.tool_calls
                ]
            
            messages.append(assistant_message)
            
            if not response.tool_calls:
                final_response = response.content
                break
                
            # Process other tool calls
            for tool_call in response.tool_calls:
                tool_result = handle_tool_call(tool_call)
                messages.append({
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": tool_result,
                    "tool_call_id": tool_call.id
                })
                print(f"Tool result: {tool_result}")
            
            iteration += 1
                
        return final_response

# Example usage
if __name__ == "__main__":
    agent = Agent()
    messages = [agent.system_prompt]
    
    print("\n=== AI Assistant ===")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        messages.append({"role": "user", "content": user_input})
        print("\n=== Processing... ===")
        result = agent.process_conversation(messages)
        
        print(f"\nAssistant: {result}")
        messages.append({"role": "assistant", "content": result})
        
    print("\nGoodbye!")
