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
        # Secure: Read from environment variable
        self.api_key = os.getenv("LLM_API_KEY")
        self.api_base = os.getenv("LLM_API_BASE", "https://api.deepseek.com") # Default to DeepSeek
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")
        
        # Professional Templates for Heuristic Mode (Chinese)
        self.templates = {
            "RISK_ANALYSIS": [
                "根据多维深度分析，**{repo_name}** 的风险评分为 **{risk_percent}/100** ({status})。这主要是由其{driver_factor}决定的。",
                "系统计算出 **{repo_name}** 的风险指数为 **{risk_percent}**。{status} 状态表明{implication}。",
                "深度依赖扫描显示风险水平为 **{risk_percent}%**。主要贡献因素包括{driver_factor}和拓扑结构脆弱性。"
            ],
            "ECOSYSTEM_INSIGHT": [
                "OpenRank 分析显示该项目的影响力得分为 **{openrank_val}**。它是开源生态中的{rank_desc}玩家。",
                "凭借 **{openrank_val}** 的 OpenRank，**{repo_name}** 展现了{rank_desc}社区影响力。活跃度水平{activity_desc}。",
            ],
            "SECURITY_ADVICE": [
                "💡 **行动建议**: 鉴于 {risk_level} 风险，我们建议{action}。特别关注{focus_area}。",
                "🛡️ **缓解策略**: {action}。图结构表明在{focus_area}存在高中介中心性节点。",
            ]
        }

    async def process_query(self, query: str, context: dict):
        """
        Smart Dispatcher: Try LLM first, fallback to Template Engine.
        """
        if self.api_key:
            try:
                # IMPORTANT: Ensure call_llm is awaited properly
                result = await self.call_llm(query, context)
                return result
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
        你是一个名为 ChainWarner AI 的专家，专注于软件供应链安全和开源生态分析。
        
        当前项目上下文:
        - 项目名称: {repo_name}
        - 风险评分: {risk_score:.2f} (0=安全, 1=危险)
        - 关键指标: {description}
        - 依赖数量: {len(context['nodes']) - 1}
        
        指令:
        - 请根据上述上下文回答用户的提问。
        - 回答必须使用**中文**。
        - 保持专业、简洁且有洞察力。
        - 关键指标数值请使用加粗显示。
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
        repo_name = root.get('name', '未知项目')
        risk = root.get('risk_score', 0.5)
        percent = round(risk * 100, 1)
        
        # Parse Description for hidden metrics (OpenRank/Constraint)
        desc = root.get('description', '') # "Constraint: 0.12 | Rank: 0.85"
        openrank_val = "未知"
        constraint_val = "未知"
        
        if "Rank:" in desc:
            try:
                # Rank: 0.85 -> Normalized score. Let's convert back to rough raw OpenRank for display
                rank_score = float(desc.split("Rank:")[1].strip())
                openrank_val = f"{rank_score * 1000:.0f}" # Reverse normalization approx
            except:
                pass
                
        # Determine Status (Chinese)
        if risk < 0.4:
            status = "安全 ✅"
            risk_level = "低"
            implication = "开发实践稳定"
            driver_factor = "高 OpenRank 和持续的活跃度"
            rank_desc = "主导型"
            activity_desc = "强劲"
            action = "保持当前的审计计划"
            focus_area = "传递性依赖"
        elif risk < 0.7:
            status = "警告 ⚠️"
            risk_level = "中等"
            implication = "潜在的结构性弱点"
            driver_factor = "复杂的依赖链"
            rank_desc = "成长型"
            activity_desc = "波动"
            action = "锁定依赖版本"
            focus_area = "直接依赖"
        else:
            status = "高危 🚨"
            risk_level = "高"
            implication = "急需安全关注"
            driver_factor = "高结构洞约束和低活跃度"
            rank_desc = "小众/边缘"
            activity_desc = "停滞"
            action = "立即进行人工代码审查"
            focus_area = "安全补丁"

        # Intent Routing
        # Use simple keyword matching for Chinese/English
        if any(w in query for w in ["risk", "score", "safe", "status", "风险", "安全", "分数"]):
            tpl = random.choice(self.templates["RISK_ANALYSIS"])
            return tpl.format(
                repo_name=repo_name, risk_percent=percent, status=status,
                driver_factor=driver_factor, implication=implication
            )
            
        elif any(w in query for w in ["rank", "influence", "community", "trend", "排名", "影响", "社区", "趋势"]):
            tpl = random.choice(self.templates["ECOSYSTEM_INSIGHT"])
            return tpl.format(
                repo_name=repo_name, openrank_val=openrank_val,
                rank_desc=rank_desc, activity_desc=activity_desc
            )
            
        elif any(w in query for w in ["fix", "advice", "suggestion", "help", "建议", "修复", "怎么办"]):
            tpl = random.choice(self.templates["SECURITY_ADVICE"])
            return tpl.format(
                risk_level=risk_level, action=action, focus_area=focus_area
            )
            
        # Default General Response
        return f"我已经分析了 **{repo_name}**。它的风险评分为 **{percent}**，OpenRank 为 **{openrank_val}**。有什么我可以进一步帮助您的吗？"
