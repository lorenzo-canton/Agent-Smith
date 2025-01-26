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
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_tasks",
            "description": "Get a markdown formatted list of all scheduled tasks",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_task_by_objective",
            "description": "Delete scheduled tasks containing specific objective text",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "Text to search for in objectives to delete"
                    }
                },
                "required": ["objective"]
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

def list_all_tasks():
    """Return all tasks in markdown format"""
    if not os.path.exists(TASKS_FILE):
        return "No scheduled tasks found"
    
    with open(TASKS_FILE, 'r') as f:
        tasks = json.load(f)
        
    if not tasks:
        return "No scheduled tasks found"
        
    markdown = ["# Scheduled Tasks\n"]
    for cron_expr, objectives in tasks.items():
        markdown.append(f"## {cron_expr}")
        markdown.extend([f"- {obj}" for obj in objectives])
        markdown.append("")  # Add empty line between groups
        
    return "\n".join(markdown)

def delete_task_by_objective(objective):
    """Delete tasks containing the objective text"""
    if not os.path.exists(TASKS_FILE):
        return "No scheduled tasks found"
        
    with open(TASKS_FILE, 'r') as f:
        tasks = json.load(f)
        
    deleted = False
    for cron_expr, objectives in list(tasks.items()):
        # Remove matching objectives
        original_count = len(objectives)
        tasks[cron_expr] = [obj for obj in objectives if objective.lower() not in obj.lower()]
        
        # If all objectives removed, delete the cron entry
        if len(tasks[cron_expr]) == 0:
            del tasks[cron_expr]
            deleted = True
        elif len(tasks[cron_expr]) != original_count:
            deleted = True
            
    if deleted:
        with open(TASKS_FILE, 'w') as f:
            json.dump(tasks, f, indent=2)
        return f"Deleted tasks containing: {objective}"
    return f"No tasks found containing: {objective}"

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
            
    if tool_call.function.name == "list_all_tasks":
        return list_all_tasks()
        
    if tool_call.function.name == "delete_task_by_objective":
        return delete_task_by_objective(args["objective"])
            
    return "Unknown tool"
