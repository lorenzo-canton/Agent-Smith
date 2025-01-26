import unittest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
from agent import Agent
from tools import tools, handle_tool_call

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()
        self.mock_response = MagicMock()
        self.mock_response.choices = [MagicMock()]
        self.mock_response.choices[0].message = MagicMock()
        
    @patch('agent.OpenAI')
    def test_agent_initialization(self, mock_openai):
        agent = Agent()
        self.assertIsNotNone(agent.client)
        self.assertIsInstance(agent.system_prompt, dict)
        self.assertEqual(agent.system_prompt['role'], 'system')
        
    @patch('agent.OpenAI')
    def test_send_messages(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = self.mock_response
        
        messages = [{'role': 'user', 'content': 'test'}]
        response = self.agent.send_messages(messages)
        
        mock_client.chat.completions.create.assert_called_once_with(
            model='gpt-3.5-turbo',
            messages=messages,
            tools=tools
        )
        self.assertIsNotNone(response)
        
    @patch('agent.Agent.send_messages')
    def test_process_conversation_direct_response(self, mock_send):
        mock_send.return_value.content = "Direct response"
        mock_send.return_value.tool_calls = None
        
        messages = [{'role': 'user', 'content': 'test'}]
        result = self.agent.process_conversation(messages)
        
        self.assertEqual(result, "Direct response")
        
    @patch('agent.Agent.send_messages')
    def test_process_conversation_with_tool_calls(self, mock_send):
        mock_send.return_value.content = None
        mock_send.return_value.tool_calls = [
            MagicMock(function=MagicMock(
                name='get_weather',
                arguments=json.dumps({'location': 'Rome, Italy'})
            ))
        ]
        
        messages = [{'role': 'user', 'content': 'test'}]
        result = self.agent.process_conversation(messages)
        
        self.assertEqual(result, "Rome, Italy: 24℃")
        
    @patch('agent.Agent.send_messages')
    def test_process_conversation_final_response(self, mock_send):
        mock_send.return_value.content = None
        mock_send.return_value.tool_calls = [
            MagicMock(function=MagicMock(
                name='final_response',
                arguments=json.dumps({'content': 'Final answer'})
            ))
        ]
        
        messages = [{'role': 'user', 'content': 'test'}]
        result = self.agent.process_conversation(messages)
        
        self.assertEqual(result, "Final answer")

class TestTools(unittest.TestCase):
    def setUp(self):
        # Clear any existing tasks before each test
        if Path("scheduled_tasks.json").exists():
            Path("scheduled_tasks.json").unlink()

    def test_handle_tool_call_weather(self):
        mock_call = MagicMock()
        mock_call.function.name = 'get_weather'
        mock_call.function.arguments = json.dumps({'location': 'Paris, France'})
        
        result = handle_tool_call(mock_call)
        self.assertEqual(result, "Paris, France: 24℃")
        
    def test_handle_tool_call_schedule(self):
        test_cases = [
            {
                "input": {
                    'task_objective': 'Test task',
                    'datetime': '2024-01-01 12:00',
                    'is_periodic': False
                },
                "expected": "Task 'Test task' scheduled for 2024-01-01 12:00 (one-time)"
            },
            {
                "input": {
                    'task_objective': 'Daily standup',
                    'datetime': '2024-01-02 09:30',
                    'is_periodic': True
                },
                "expected": "Task 'Daily standup' scheduled for 2024-01-02 09:30 (recurring)"
            }
        ]
        
        for case in test_cases:
            mock_call = MagicMock()
            mock_call.function.name = 'schedule_task'
            mock_call.function.arguments = json.dumps(case["input"])
            
            result = handle_tool_call(mock_call)
            self.assertEqual(result, case["expected"])
            
            # Verify task was saved
            tasks = json.loads(Path("scheduled_tasks.json").read_text())
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]['task_objective'], case["input"]['task_objective'])
            Path("scheduled_tasks.json").unlink()  # Clean up after each test

    def test_handle_tool_call_schedule_invalid(self):
        mock_call = MagicMock()
        mock_call.function.name = 'schedule_task'
        mock_call.function.arguments = json.dumps({
            'task_objective': 'Invalid date',
            'datetime': 'invalid-date',
            'is_periodic': False
        })
        
        result = handle_tool_call(mock_call)
        self.assertTrue("Error scheduling task" in result)
        self.assertFalse(Path("scheduled_tasks.json").exists())
        
    def test_handle_tool_call_unknown(self):
        mock_call = MagicMock()
        mock_call.function.name = 'unknown_tool'
        mock_call.function.arguments = json.dumps({})
        
        result = handle_tool_call(mock_call)
        self.assertEqual(result, "Unknown tool")

if __name__ == '__main__':
    unittest.main()
