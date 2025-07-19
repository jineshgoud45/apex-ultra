import os
from datetime import datetime
from typing import List, Dict, Optional
import aiohttp

class TraderEngine:
    """
    Modular agent for autonomous trading and portfolio management.
    Analyzes markets, executes trades, manages portfolios; integrates with AGI/LLM; supports human-in-the-loop workflows, explainability, compliance, and auditability.
    """
    def __init__(self, agi_brain=None, api_key=None, endpoint=None):
        self.agi_brain = agi_brain
        self.api_key = api_key or os.getenv("GPT25PRO_API_KEY")
        self.endpoint = endpoint or "https://api.gpt25pro.example.com/v1/generate"
        self.trade_log = []
        self.plugins = []  # For pluggable trading APIs, strategies, etc.
        self.portfolio = {}

    def register_plugin(self, plugin):
        self.plugins.append(plugin)

    async def analyze_market(self, market: str, analysis_instructions: str = "", user_id: Optional[str] = None) -> Dict:
        """Analyze a market using LLM or plugins."""
        analysis = None
        for plugin in self.plugins:
            if hasattr(plugin, 'analyze_market'):
                analysis = await plugin.analyze_market(market, analysis_instructions)
                break
        if not analysis:
            prompt = f"Analyze the {market} market. {analysis_instructions}"
            try:
                async with aiohttp.ClientSession() as session:
                    response = await session.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"prompt": prompt, "max_tokens": 512}
                    )
                    data = await response.json()
                    analysis = data.get("text", "")
            except Exception as e:
                analysis = f"[Error: {str(e)}]"
        analysis_result = {
            "market": market,
            "instructions": analysis_instructions,
            "analysis": analysis,
            "analyst": user_id or "AI",
            "timestamp": datetime.now().isoformat(),
            "status": "pending_review"
        }
        self.trade_log.append(analysis_result)
        return analysis_result

    async def execute_trade(self, symbol: str, action: str, amount: float, user_id: Optional[str] = None) -> Dict:
        """Execute a trade (buy/sell) using LLM or plugins."""
        trade_suggestion = None
        for plugin in self.plugins:
            if hasattr(plugin, 'execute_trade'):
                trade_suggestion = await plugin.execute_trade(symbol, action, amount)
                break
        if not trade_suggestion:
            prompt = f"Should I {action} {amount} of {symbol}? Justify the decision."
            try:
                async with aiohttp.ClientSession() as session:
                    response = await session.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"prompt": prompt, "max_tokens": 256}
                    )
                    data = await response.json()
                    trade_suggestion = data.get("text", "")
            except Exception as e:
                trade_suggestion = f"[Error: {str(e)}]"
        # Simulate portfolio update
        if action == "buy":
            self.portfolio[symbol] = self.portfolio.get(symbol, 0) + amount
        elif action == "sell":
            self.portfolio[symbol] = max(0, self.portfolio.get(symbol, 0) - amount)
        trade_result = {
            "symbol": symbol,
            "action": action,
            "amount": amount,
            "suggestion": trade_suggestion,
            "trader": user_id or "AI",
            "timestamp": datetime.now().isoformat(),
            "status": "pending_review"
        }
        self.trade_log.append(trade_result)
        return trade_result

    def review_trade(self, trade_id: int, approve: bool, reviewer_id: Optional[str] = None, comments: Optional[str] = None) -> Dict:
        """Review and approve/reject a trade."""
        if trade_id >= len(self.trade_log):
            return {"error": "Trade not found"}
        self.trade_log[trade_id]["reviewed_by"] = reviewer_id
        self.trade_log[trade_id]["review_comments"] = comments
        self.trade_log[trade_id]["status"] = "approved" if approve else "rejected"
        return self.trade_log[trade_id]

    def explain_trade(self, trade_id: int) -> str:
        """Explain the rationale and process for a given trade."""
        if trade_id >= len(self.trade_log):
            return "Trade not found"
        trade = self.trade_log[trade_id]
        explanation = (
            f"Trade executed on {trade['timestamp']} by {trade['trader']}. "
            f"Action: {trade.get('action', 'N/A')} {trade.get('amount', 'N/A')} of {trade.get('symbol', 'N/A')}. "
            f"AI suggestion: {trade.get('suggestion', 'N/A')}. "
            f"Status: {trade['status']}."
        )
        if "reviewed_by" in trade:
            explanation += f" Reviewed by {trade['reviewed_by']} with comments: {trade.get('review_comments', '')}."
        return explanation

    def get_portfolio(self) -> Dict:
        """Return the current portfolio holdings."""
        return self.portfolio

    def audit_log(self) -> List[Dict]:
        """Return the full audit log of all trade actions."""
        return self.trade_log 