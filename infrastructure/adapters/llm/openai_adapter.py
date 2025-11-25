"""Adapter to provide analysis/feedback using OpenAI (stub).

Replace with real SDK calls and secrets management.
"""

class OpenAIAdapter:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def generate_feedback(self, analysis_report: dict) -> str:
        # TODO: call OpenAI or other LLM provider
        return "Feedback placeholder"
