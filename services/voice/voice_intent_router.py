import re

class VoiceIntentRouter:
    # Irreversible actions that MUST be refused in voice context
    ACTION_KEYWORDS = ["buy", "purchase", "pay", "checkout", "refund", "cancel order", 
                       "password", "2fa", "delete account", "admin", "transfer"]
                       
    LIVE_KEYWORDS = ["live", "current position", "currently", "happening now", "weather now", "gap to"]
    
    @staticmethod
    def route_intent(transcript: str) -> str:
        text = transcript.lower()
        
        # 1. Action Refusal
        if any(keyword in text for keyword in VoiceIntentRouter.ACTION_KEYWORDS):
            return "action_refusal"
            
        # 2. Live Race
        if any(keyword in text for keyword in VoiceIntentRouter.LIVE_KEYWORDS):
            return "live_race"
            
        # 3. Default to RAG/historical
        return "general_rag"
