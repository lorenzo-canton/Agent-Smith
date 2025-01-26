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
            You are an autonomous assistant that executes tasks directly using available tools. When receiving requests:
            1. Analyze the request and determine if tools are needed
            2. Execute necessary tools automatically without asking for confirmation
            3. ALWAYS use final_response tool to deliver the final answer
            
            Available tools:
            - get_weather: Get weather conditions
            - schedule_task: Schedule tasks using cron expressions
            - list_all_tasks: Get markdown formatted list of all tasks
            - delete_task_by_objective: Delete tasks containing specific text
            
            For scheduling tasks:
            - If cron expression is not provided, use default "0 9 * * 1-5" (weekdays at 9am)
            - If objectives are not fully specified, make reasonable assumptions
            - Use schedule_task tool with format:
            {
                "cron_expression": "the_cron_expression",
                "objectives": ["objective1", "objective2"]
            }
            
            Guidelines:
            1. Execute tasks immediately without asking for confirmation
            2. Make reasonable assumptions when information is incomplete
            3. Always format final response for end-user
            4. Include execution results in the final response
            5. If errors occur, include them in the final response
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
