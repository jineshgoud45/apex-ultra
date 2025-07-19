import os
from datetime import datetime
from typing import List, Dict, Optional
import aiohttp

class SoftwareDeveloperEngine:
    """
    Modular agent for autonomous software development.
    Generates, reviews, and improves code; manages projects; integrates with AGI/LLM; supports human-in-the-loop workflows and auditability.
    """
    def __init__(self, agi_brain=None, api_key=None, endpoint=None):
        self.agi_brain = agi_brain
        self.api_key = api_key or os.getenv("GPT25PRO_API_KEY")
        self.endpoint = endpoint or "https://api.gpt25pro.example.com/v1/generate"
        self.project_log = []
        self.plugins = []  # For pluggable tools (linters, CI/CD, etc.)

    def register_plugin(self, plugin):
        self.plugins.append(plugin)

    async def generate_code(self, requirements: str, language: str = "python", user_id: Optional[str] = None) -> Dict:
        """Generate code from requirements using LLM or plugins."""
        suggestion = None
        # Try plugins first
        for plugin in self.plugins:
            if hasattr(plugin, 'generate_code'):
                suggestion = await plugin.generate_code(requirements, language)
                break
        if not suggestion:
            # Fallback to LLM
            prompt = f"Write {language} code for the following requirements: {requirements}"
            try:
                async with aiohttp.ClientSession() as session:
                    response = await session.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"prompt": prompt, "max_tokens": 512}
                    )
                    data = await response.json()
                    suggestion = data.get("text", "")
            except Exception as e:
                suggestion = f"[Error: {str(e)}]"
        code_result = {
            "requirements": requirements,
            "language": language,
            "suggestion": suggestion,
            "author": user_id or "AI",
            "timestamp": datetime.now().isoformat(),
            "status": "pending_review"
        }
        self.project_log.append(code_result)
        return code_result

    async def review_code(self, code: str, review_instructions: str = "", user_id: Optional[str] = None) -> Dict:
        """Review code using LLM or plugins."""
        review = None
        for plugin in self.plugins:
            if hasattr(plugin, 'review_code'):
                review = await plugin.review_code(code, review_instructions)
                break
        if not review:
            prompt = f"Review the following code and provide feedback. {review_instructions}\nCode:\n{code}"
            try:
                async with aiohttp.ClientSession() as session:
                    response = await session.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"prompt": prompt, "max_tokens": 512}
                    )
                    data = await response.json()
                    review = data.get("text", "")
            except Exception as e:
                review = f"[Error: {str(e)}]"
        review_result = {
            "code": code,
            "review_instructions": review_instructions,
            "review": review,
            "reviewer": user_id or "AI",
            "timestamp": datetime.now().isoformat(),
            "status": "pending_review"
        }
        self.project_log.append(review_result)
        return review_result

    def explain_code(self, code: str) -> str:
        """Explain code logic and structure using LLM API."""
        prompt = f"Explain the following code in detail:\n{code}"
        try:
            import asyncio
            async def get_explanation():
                async with aiohttp.ClientSession() as session:
                    response = await session.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"prompt": prompt, "max_tokens": 256}
                    )
                    data = await response.json()
                    return data.get("text", "")
            return asyncio.run(get_explanation())
        except Exception as e:
            return f"[Error: {str(e)}]"

    def audit_log(self) -> List[Dict]:
        """Return the full audit log of all code generation and review actions."""
        return self.project_log 