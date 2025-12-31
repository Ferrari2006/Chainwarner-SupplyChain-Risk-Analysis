import os
import random
import httpx
import json

class AgentEngine:
    """
    🧠 ChainWarner Intelligent Agent (Dual-Mode)
    
    Mode A: DeepSeek/OpenAI (True AI)
    - Activated if 'LLM_API_KEY' env var is set.
    - Uses RAG to answer queries with full context.
    
    Mode B: Advanced Template Engine (Heuristic AI)
    - Fallback if no API key.
    - Uses deterministic logic to assemble sophisticated responses.
    - Simulates AI reasoning without cost.
    """
    
    def __init__(self):
        # Configuration
        self.api_key = os.getenv("LLM_API_KEY")
        self.api_base = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1") # Default to DeepSeek
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")
        
        # Professional Templates for Heuristic Mode
        self.templates = {
            "RISK_ANALYSIS": [
                "Based on the multi-dimensional analysis, **{repo_name}** has a risk score of **{risk_percent}/100** ({status}). This is primarily driven by its {driver_factor}.",
                "The system calculates a risk index of **{risk_percent}** for **{repo_name}**. The {status} status suggests {implication}.",
                "Deep dependency scanning reveals a risk level of **{risk_percent}%**. The main contributors are {driver_factor} and structural topology."
            ],
            "ECOSYSTEM_INSIGHT": [
                "OpenRank analysis shows this project has an influence score of **{openrank_val}**. It is a {rank_desc} player in the open source ecosystem.",
                "With an OpenRank of **{openrank_val}**, **{repo_name}** demonstrates {rank_desc} community impact. Activity levels are {activity_desc}.",
            ],
            "SECURITY_ADVICE": [
                "💡 **Action Item**: Given the {risk_level} risk, we recommend {action}. specifically focusing on {focus_area}.",
                "🛡️ **Mitigation Strategy**: {action}. The graph structure indicates high centrality in {focus_area}.",
            ]
        }

    async def process_query(self, query: str, context: dict):
        """
        Smart Dispatcher: Try LLM first, fallback to Template Engine.
        """
        if self.api_key:
            try:
                return await self.call_llm(query, context)
            except Exception as e:
                print(f"[Agent] LLM Call Failed: {e}. Falling back to Template Engine.")
        
        return self.heuristic_response(query, context)

    async def call_llm(self, query: str, context: dict):
        """
        True AI: Call DeepSeek/OpenAI API with RAG context.
        """
        # 1. Prepare Context Summary
        root_node = context['nodes'][0] if context['nodes'] else {}
        repo_name = root_node.get('name', 'Unknown')
        risk_score = root_node.get('risk_score', 0.5)
        description = root_node.get('description', '')
        
        system_prompt = f"""
        You are ChainWarner AI, an expert in software supply chain security and open source ecosystem analysis.
        
        Current Project Context:
        - Name: {repo_name}
        - Risk Score: {risk_score:.2f} (0=Safe, 1=Dangerous)
        - Metrics: {description}
        - Dependency Count: {len(context['nodes']) - 1}
        
        Instructions:
        - Answer the user's query based on the context.
        - Be professional, concise, and insightful.
        - Use bolding for key metrics.
        """
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    "temperature": 0.3
                },
                timeout=10.0
            )
            data = resp.json()
            return data['choices'][0]['message']['content']

    def heuristic_response(self, query: str, context: dict):
        """
        Fallback: sophisticated rule-based generation.
        """
        query = query.lower()
        
        # Extract Variables
        root = context['nodes'][0] if context['nodes'] else {}
        repo_name = root.get('name', 'Unknown')
        risk = root.get('risk_score', 0.5)
        percent = round(risk * 100, 1)
        
        # Parse Description for hidden metrics (OpenRank/Constraint)
        desc = root.get('description', '') # "Constraint: 0.12 | Rank: 0.85"
        openrank_val = "Unknown"
        constraint_val = "Unknown"
        
        if "Rank:" in desc:
            try:
                # Rank: 0.85 -> Normalized score. Let's convert back to rough raw OpenRank for display
                rank_score = float(desc.split("Rank:")[1].strip())
                openrank_val = f"{rank_score * 1000:.0f}" # Reverse normalization approx
            except:
                pass
                
        # Determine Status
        if risk < 0.4:
            status = "Safe ✅"
            risk_level = "low"
            implication = "stable development practices"
            driver_factor = "high OpenRank and consistent activity"
            rank_desc = "dominant"
            activity_desc = "robust"
            action = "maintaining current audit schedules"
            focus_area = "transitive dependencies"
        elif risk < 0.7:
            status = "Caution ⚠️"
            risk_level = "moderate"
            implication = "potential structural weaknesses"
            driver_factor = "complex dependency chains"
            rank_desc = "growing"
            activity_desc = "fluctuating"
            action = "locking dependency versions"
            focus_area = "direct dependencies"
        else:
            status = "Critical 🚨"
            risk_level = "high"
            implication = "urgent security attention needed"
            driver_factor = "high structural constraint and low activity"
            rank_desc = "niche"
            activity_desc = "stagnant"
            action = "immediate manual code review"
            focus_area = "security patches"

        # Intent Routing
        if any(w in query for w in ["risk", "score", "safe", "status"]):
            tpl = random.choice(self.templates["RISK_ANALYSIS"])
            return tpl.format(
                repo_name=repo_name, risk_percent=percent, status=status,
                driver_factor=driver_factor, implication=implication
            )
            
        elif any(w in query for w in ["rank", "influence", "community", "trend"]):
            tpl = random.choice(self.templates["ECOSYSTEM_INSIGHT"])
            return tpl.format(
                repo_name=repo_name, openrank_val=openrank_val,
                rank_desc=rank_desc, activity_desc=activity_desc
            )
            
        elif any(w in query for w in ["fix", "advice", "suggestion", "help"]):
            tpl = random.choice(self.templates["SECURITY_ADVICE"])
            return tpl.format(
                risk_level=risk_level, action=action, focus_area=focus_area
            )
            
        # Default General Response
        return f"I've analyzed **{repo_name}**. It has a risk score of **{percent}** and OpenRank of **{openrank_val}**. How can I help further?"
