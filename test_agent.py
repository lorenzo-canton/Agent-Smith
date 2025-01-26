import unittest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
from agent import Agent
from tools import tools, handle_tool_call, TASKS_FILE

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
        if Path(TASKS_FILE).exists():
            Path(TASKS_FILE).unlink()

    def test_handle_tool_call_weather(self):
        mock_call = MagicMock()
        mock_call.function.name = 'get_weather'
        mock_call.function.arguments = json.dumps({'location': 'Paris, France'})
        
        result = handle_tool_call(mock_call)
        self.assertEqual(result, "Paris, France: 24℃")
        
    def test_handle_tool_call_schedule(self):
        mock_call = MagicMock()
        mock_call.function.name = 'schedule_task'
        mock_call.function.arguments = json.dumps({
            'cron_expression': '0 9 * * 1-5',
            'objectives': ['Daily standup']
        })
        
        result = handle_tool_call(mock_call)
        self.assertEqual(result, "Tasks scheduled with cron expression: 0 9 * * 1-5")
        
        # Verify task was saved
        with open(TASKS_FILE, 'r') as f:
            tasks = json.load(f)
            self.assertIn('0 9 * * 1-5', tasks)
            self.assertEqual(tasks['0 9 * * 1-5'], ['Daily standup'])

    def test_handle_tool_call_schedule_multiple_objectives(self):
        mock_call = MagicMock()
        mock_call.function.name = 'schedule_task'
        mock_call.function.arguments = json.dumps({
            'cron_expression': '0 12 * * *',
            'objectives': ['Lunch break', 'Team sync']
        })
        
        result = handle_tool_call(mock_call)
        self.assertEqual(result, "Tasks scheduled with cron expression: 0 12 * * *")
        
        # Verify tasks were saved
        with open(TASKS_FILE, 'r') as f:
            tasks = json.load(f)
            self.assertEqual(tasks['0 12 * * *'], ['Lunch break', 'Team sync'])

    def test_handle_tool_call_list_all_tasks(self):
        # First add some test data
        with open(TASKS_FILE, 'w') as f:
            json.dump({
                '0 9 * * 1-5': ['Daily standup'],
                '0 12 * * *': ['Lunch break']
            }, f)
        
        mock_call = MagicMock()
        mock_call.function.name = 'list_all_tasks'
        mock_call.function.arguments = '{}'
        
        result = handle_tool_call(mock_call)
        expected_output = """# Scheduled Tasks

## 0 9 * * 1-5
- Daily standup

## 0 12 * * *
- Lunch break
"""
        self.assertEqual(result, expected_output)

    def test_handle_tool_call_unknown(self):
        mock_call = MagicMock()
        mock_call.function.name = 'unknown_tool'
        mock_call.function.arguments = json.dumps({})
        
        result = handle_tool_call(mock_call)
        self.assertEqual(result, "Unknown tool")

if __name__ == '__main__':
    unittest.main()
