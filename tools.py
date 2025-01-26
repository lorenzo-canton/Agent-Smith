import json
import os
from datetime import datetime

TASKS_FILE = "scheduled_tasks.json"

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
            "description": "Schedule a task using cron expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "cron_expression": {
                        "type": "string",
                        "description": "Cron expression for scheduling (e.g., '0 9 * * 1-5' for weekdays at 9am)"
                    },
                    "objectives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of objectives to schedule"
                    }
                },
                "required": ["cron_expression", "objectives"]
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_scheduled_tasks",
            "description": "Get list of all scheduled tasks",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

def save_task(task_data):
    """Save scheduled task to JSON file"""
    try:
        # Validate cron expression format
        cron_parts = task_data["cron_expression"].split()
        if len(cron_parts) != 5:
            raise ValueError("Invalid cron expression - must have 5 parts")
            
        # Create file if it doesn't exist
        if not os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, 'w') as f:
                json.dump({}, f)
                
        # Read existing tasks
        with open(TASKS_FILE, 'r') as f:
            tasks = json.load(f)
            
        # Add new task
        cron_expr = task_data["cron_expression"]
        if cron_expr not in tasks:
            tasks[cron_expr] = []
        tasks[cron_expr].extend(task_data["objectives"])
        
        # Save updated list
        with open(TASKS_FILE, 'w') as f:
            json.dump(tasks, f, indent=2)
            
        return True
    except Exception as e:
        print(f"Error saving task: {e}")
        return False

def handle_tool_call(tool_call):
    args = json.loads(tool_call.function.arguments)
    print(f"\n=== Handling {tool_call.function.name} ===")
    print(f"Arguments: {args}")
    
    if tool_call.function.name == "get_weather":
        return f"{args['location']}: 24℃"
    
    if tool_call.function.name == "schedule_task":
        try:
            task_data = {
                "cron_expression": args["cron_expression"],
                "objectives": args["objectives"]
            }
            save_task(task_data)
            return f"Tasks scheduled with cron expression: {args['cron_expression']}"
        except Exception as e:
            return f"Error scheduling task: {str(e)}"
    
    if tool_call.function.name == "get_scheduled_tasks":
        try:
            if os.path.exists(TASKS_FILE):
                with open(TASKS_FILE, 'r') as f:
                    tasks = json.load(f)
                # Format the output for better readability
                formatted_tasks = []
                for cron_expr, objectives in tasks.items():
                    formatted_tasks.append(f"Cron: {cron_expr}")
                    formatted_tasks.extend([f" - {obj}" for obj in objectives])
                return "\n".join(formatted_tasks)
            return "No scheduled tasks found"
        except Exception as e:
            return f"Error reading tasks: {str(e)}"
            
    return "Unknown tool"
