import re
import random

class AgentEngine:
    """
    🤖 ChainWarner Analytics Agent (Lite RAG)
    Uses regex-based intent recognition and context-aware templates 
    to answer user questions about the current project analysis.
    """
    
    def __init__(self):
        self.intents = {
            "RISK_SCORE": [r"risk", r"score", r"danger", r"safe", r"status"],
            "DEPENDENCIES": [r"depend", r"lib", r"package", r"import"],
            "MAINTAINER": [r"maintain", r"author", r"owner", r"who"],
            "VULNERABILITY": [r"vuln", r"cve", r"exploit", r"hack", r"bug"],
            "ADVICE": [r"fix", r"advice", r"suggestion", r"todo", r"help"]
        }

    def process_query(self, query: str, context: dict):
        """
        Process user query with graph context.
        context: The JSON result from /graph/{owner}/{repo}
        """
        query = query.lower()
        
        # 1. Detect Intent
        detected_intent = None
        for intent, keywords in self.intents.items():
            if any(k in query for k in keywords):
                detected_intent = intent
                break
        
        if not detected_intent:
            return "I'm sorry, I didn't quite understand that. You can ask me about the risk score, dependencies, or security advice."

        # 2. Extract Context Variables
        root_node = context['nodes'][0] if context['nodes'] else {}
        repo_name = root_node.get('name', 'Unknown')
        risk_score = root_node.get('risk_score', 0.5)
        risk_percent = round(risk_score * 100, 1)
        
        # 3. Generate Response (RAG-Lite)
        if detected_intent == "RISK_SCORE":
            status = "Safe ✅" if risk_score < 0.4 else ("Caution ⚠️" if risk_score < 0.7 else "Critical 🚨")
            return f"The current risk score for **{repo_name}** is **{risk_percent}/100**. Status: {status}. This is calculated based on activity, CVEs, and dependency structure."

        elif detected_intent == "DEPENDENCIES":
            dep_count = len(context['nodes']) - 1
            top_deps = [n['name'] for n in context['nodes'][1:4]]
            return f"This project has **{dep_count}** direct dependencies analyzed. Key dependencies include: {', '.join(top_deps)}. A complex dependency tree increases the attack surface."

        elif detected_intent == "VULNERABILITY":
            if risk_score > 0.7:
                return f"🚨 **High Vulnerability Alert**: We detected potential issues in the dependency graph structure (High Constraint). EasyGraph analysis suggests this project is a 'choke point' in the supply chain."
            else:
                return "✅ No critical vulnerabilities detected in the current snapshot. However, keep monitoring for new CVEs."

        elif detected_intent == "ADVICE":
            if risk_score > 0.5:
                return "💡 **Recommendation**: Consider pinning dependency versions, reviewing the changelogs of 'orange' nodes, and setting up automated alerts."
            else:
                return "💡 **Recommendation**: Keep up the good work! Regular audits are still recommended."
        
        return "I can analyze that for you."
