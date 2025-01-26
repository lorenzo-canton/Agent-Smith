import json
from datetime import datetime

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather conditions for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and country, e.g. Rome, Italy"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": "Schedule a reminder or task to execute at specific time",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_objective": {
                        "type": "string", 
                        "description": "Detailed description of the task to remember"
                    },
                    "datetime": {
                        "type": "string", 
                        "description": "Date and time in YYYY-MM-DD HH:MM format, must be in the future"
                    },
                    "is_periodic": {
                        "type": "boolean", 
                        "description": "Whether the reminder should repeat daily at the same time"
                    }
                },
                "required": ["task_objective", "datetime", "is_periodic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_response",
            "description": "Delivers final answer to user",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Final response content"}
                },
                "required": ["content"]
            }
        }
    }
]

def handle_tool_call(tool_call):
    args = json.loads(tool_call.function.arguments)
    print(f"\n=== Handling {tool_call.function.name} ===")
    print(f"Arguments: {args}")
    
    if tool_call.function.name == "get_weather":
        return f"{args['location']}: 24℃"
    
    if tool_call.function.name == "schedule_task":
        try:
            task_time = datetime.strptime(args['datetime'], "%Y-%m-%d %H:%M")
            periodicity = "recurring" if args['is_periodic'] else "one-time"
            return f"Task '{args['task_objective']}' scheduled for {task_time.strftime('%Y-%m-%d %H:%M')} ({periodicity})"
        except ValueError as e:
            return f"Error scheduling task: {str(e)}"
    
    return "Unknown tool"
