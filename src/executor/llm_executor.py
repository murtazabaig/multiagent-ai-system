from typing import Dict, Any
import logging

class LLMExecutor:
    """
    An LLM-based executor that performs reasoning over context and synthesizes the final answer.
    """
    def __init__(self, model_name: str = "google/flan-t5-base"):
        print(f"Initializing LLMExecutor with model: {model_name}")
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logging.getLogger("transformers").setLevel(logging.ERROR)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model_loaded = True
        except ImportError:
            print("Warning: transformers library not found. Running in fallback mode.")
            self.model_loaded = False
        except Exception as e:
            print(f"Warning: Failed to load model {model_name} due to: {e}. Running in fallback mode.")
            self.model_loaded = False

    def reason(self, query: str, context: str) -> str:
        """Analyze the retrieved context to gather facts before final execution."""
        if not self.model_loaded:
            return "Fallback reasoning: analyzed context and found potential answers."
            
        # Ensure context isn't too long for the small model (flan-t5-small handles ~512 tokens well)
        short_context = context[:1000] if context else "No context retrieved."
        prompt = f"Context: {short_context}\nAnalyze the context to answer the query: {query}\nReasoning:"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=100
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            return f"Reasoning failed: {e}"

    def execute(self, query: str, context: str) -> Dict[str, Any]:
        """Synthesize the final execution/answer based on reasoning and context."""
        if not self.model_loaded:
            return {"result": f"Fallback execution: simulated action '{query}'"}
            
        short_context = context[:1500] if context else "No context available."
        prompt = f"System: You are an intelligent reasoning AI. Based on the following context, provide a detailed and well-formatted explanation to answer the user's question. Do not just blindly copy the text or titles. Explain the concepts fully in your own words.\n\nContext: {short_context}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=150,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"result": response}
        except Exception as e:
            return {"error": str(e), "result": "Execution failed."}
