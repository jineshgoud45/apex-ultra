import os
import aiohttp
from datetime import datetime
from typing import List, Dict, Optional

class ConsultantEngine:
    """
    Modular agent for business, marketing, and technical consulting.
    Provides research, reports, and recommendations; integrates with AGI/LLM; supports human-in-the-loop workflows, explainability, and auditability.
    """
    def __init__(self, agi_brain=None, api_key=None, endpoint=None):
        self.agi_brain = agi_brain
        self.api_key = api_key or os.getenv("GPT25PRO_API_KEY")
        self.endpoint = endpoint or "https://api.gpt25pro.example.com/v1/generate"
        self.consulting_log = []
        self.plugins = []  # For pluggable tools, data sources, templates, etc.

    def register_plugin(self, plugin):
        self.plugins.append(plugin)

    async def consult(self, topic: str, instructions: str = "", user_id: Optional[str] = None) -> Dict:
        """Provide consulting (research, analysis, recommendations) using LLM or plugins."""
        report = None
        for plugin in self.plugins:
            if hasattr(plugin, 'consult'):
                report = await plugin.consult(topic, instructions)
                break
        if not report:
            prompt = f"Provide a consulting report on: {topic}. {instructions}"
            try:
                async with aiohttp.ClientSession() as session:
                    response = await session.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"prompt": prompt, "max_tokens": 512}
                    )
                    data = await response.json()
                    report = data.get("text", "")
            except Exception as e:
                report = f"[Error: {str(e)}]"
        consult_result = {
            "topic": topic,
            "instructions": instructions,
            "report": report,
            "consultant": user_id or "AI",
            "timestamp": datetime.now().isoformat(),
            "status": "pending_review"
        }
        self.consulting_log.append(consult_result)
        return consult_result

    def review_report(self, report_id: int, approve: bool, reviewer_id: Optional[str] = None, comments: Optional[str] = None) -> Dict:
        """Review and approve/reject a consulting report."""
        if report_id >= len(self.consulting_log):
            return {"error": "Report not found"}
        self.consulting_log[report_id]["reviewed_by"] = reviewer_id
        self.consulting_log[report_id]["review_comments"] = comments
        self.consulting_log[report_id]["status"] = "approved" if approve else "rejected"
        return self.consulting_log[report_id]

    def explain_report(self, report_id: int) -> str:
        """Explain the rationale and process for a given consulting report."""
        if report_id >= len(self.consulting_log):
            return "Report not found"
        report = self.consulting_log[report_id]
        explanation = (
            f"Consulting report generated on {report['timestamp']} by {report['consultant']}. "
            f"Topic: {report.get('topic', 'N/A')}. "
            f"Instructions: {report.get('instructions', 'N/A')}. "
            f"Status: {report['status']}.")
        if "reviewed_by" in report:
            explanation += f" Reviewed by {report['reviewed_by']} with comments: {report.get('review_comments', '')}."
        return explanation

    def audit_log(self) -> List[Dict]:
        """Return the full audit log of all consulting actions."""
        return self.consulting_log 