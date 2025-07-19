from typing import Any, Dict

class UnifiedAGIInterface:
    """
    Unified, multimodal, iterative AGI interface for text, image, video, and audio.
    Maintains session/context, routes input, and provides explanations for all outputs.
    """
    def __init__(self, text_module, image_module, video_module, audio_module):
        self.text_module = text_module
        self.image_module = image_module
        self.video_module = video_module
        self.audio_module = audio_module
        self.session_context: Dict[str, Any] = {}

    def handle_input(self, user_input: Any, modality: str = "auto") -> Dict[str, Any]:
        if modality == "auto":
            modality = self.detect_modality(user_input)
        if modality == "text":
            result = self.text_module.process(user_input, self.session_context)
        elif modality == "image":
            result = self.image_module.process(user_input, self.session_context)
        elif modality == "video":
            result = self.video_module.process(user_input, self.session_context)
        elif modality == "audio":
            result = self.audio_module.process(user_input, self.session_context)
        else:
            raise ValueError("Unknown modality")
        explanation = result.get("explanation") or self.explain_output(result, modality)
        self.session_context["last_result"] = result
        return {"result": result, "explanation": explanation}

    def detect_modality(self, user_input: Any) -> str:
        if isinstance(user_input, str):
            return "text"
        elif isinstance(user_input, bytes):
            return "audio"
        elif hasattr(user_input, "shape"):
            return "image"
        return "text"

    def explain_output(self, result: Any, modality: str) -> str:
        module = getattr(self, f"{modality}_module", None)
        if module and hasattr(module, "explain_output"):
            return module.explain_output(result)
        return "No explanation available." 