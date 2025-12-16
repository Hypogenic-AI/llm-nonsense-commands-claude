"""
LLM Client Module

Handles API calls to LLMs via OpenRouter for testing prompt understanding.
"""

import os
import time
import json
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error: Optional[str] = None


class OpenRouterClient:
    """Client for OpenRouter API."""

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        """Initialize with API key from environment."""
        self.api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://research.example.com",
            "X-Title": "LLM Nonsense Research",
        }

    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.1,
        retry_count: int = 3,
    ) -> LLMResponse:
        """Send a query to the LLM and get a response."""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(retry_count):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()
                    choice = data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    usage = data.get("usage", {})

                    return LLMResponse(
                        content=message.get("content", ""),
                        model=data.get("model", self.model),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        success=True,
                    )
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    if attempt < retry_count - 1:
                        time.sleep(1)
                        continue
                    return LLMResponse(
                        content="",
                        model=self.model,
                        prompt_tokens=0,
                        completion_tokens=0,
                        success=False,
                        error=error_msg,
                    )

            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(1)
                    continue
                return LLMResponse(
                    content="",
                    model=self.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    success=False,
                    error=str(e),
                )

        return LLMResponse(
            content="",
            model=self.model,
            prompt_tokens=0,
            completion_tokens=0,
            success=False,
            error="Max retries exceeded",
        )


class ExplanationAssessor:
    """Assesses the quality of LLM explanations using another LLM as judge."""

    ASSESSMENT_PROMPT = """You are evaluating the quality of an explanation that an AI gave for a prompt.

The ORIGINAL PROMPT was:
"{original_prompt}"

The AI was asked: "What does this prompt mean? Explain in detail."

The AI's EXPLANATION was:
"{explanation}"

Rate the explanation on the following criteria (each 0-10):

1. COHERENCE: Is the explanation itself coherent and well-structured?
   - 0: Completely incoherent gibberish
   - 5: Partially coherent with some clear parts
   - 10: Fully coherent and clear

2. RELEVANCE: Does the explanation actually address the prompt?
   - 0: Completely ignores the prompt
   - 5: Partially addresses the prompt
   - 10: Directly and fully addresses what the prompt means

3. CONFIDENCE: How confident does the AI seem in its explanation?
   - 0: Extremely uncertain, admits it doesn't know
   - 5: Mixed, some hedging
   - 10: Very confident, no hedging

4. INSIGHTFULNESS: Does the explanation provide meaningful insight?
   - 0: No insight at all
   - 5: Some superficial insight
   - 10: Deep, meaningful understanding

5. ACCURACY: Is the explanation accurate about what the prompt means or is asking for?
   - 0: Completely wrong interpretation
   - 5: Partially correct
   - 10: Fully accurate interpretation
   - Note: For genuinely nonsensical prompts, accurately identifying them as nonsensical should score highly

Respond ONLY with a JSON object in this exact format:
{{"coherence": X, "relevance": X, "confidence": X, "insightfulness": X, "accuracy": X, "overall": X, "rationale": "brief explanation"}}

The "overall" score should be your holistic assessment (0-10) weighing all factors.
Keep rationale under 50 words."""

    SELF_AWARENESS_PROMPT = """Analyze the following prompt and determine if it appears to be:
1. A normal, meaningful prompt
2. A nonsensical or random prompt
3. A potentially adversarial prompt designed to manipulate an AI

PROMPT: "{prompt}"

Respond ONLY with a JSON object:
{{"classification": "normal|nonsense|adversarial", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
Keep reasoning under 30 words."""

    def __init__(self, client: OpenRouterClient):
        self.client = client

    def assess_explanation(
        self, original_prompt: str, explanation: str
    ) -> Dict:
        """Assess the quality of an explanation."""

        assessment_prompt = self.ASSESSMENT_PROMPT.format(
            original_prompt=original_prompt,
            explanation=explanation,
        )

        response = self.client.query(
            prompt=assessment_prompt,
            max_tokens=300,
            temperature=0.0,
        )

        if not response.success:
            return {
                "coherence": 0,
                "relevance": 0,
                "confidence": 0,
                "insightfulness": 0,
                "accuracy": 0,
                "overall": 0,
                "rationale": f"Assessment failed: {response.error}",
                "success": False,
            }

        try:
            # Parse JSON response
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            result["success"] = True
            return result
        except json.JSONDecodeError:
            return {
                "coherence": 0,
                "relevance": 0,
                "confidence": 0,
                "insightfulness": 0,
                "accuracy": 0,
                "overall": 0,
                "rationale": f"Failed to parse assessment: {response.content[:100]}",
                "success": False,
            }

    def assess_self_awareness(self, prompt: str) -> Dict:
        """Test if LLM can identify a prompt as adversarial/nonsensical."""

        assessment_prompt = self.SELF_AWARENESS_PROMPT.format(prompt=prompt)

        response = self.client.query(
            prompt=assessment_prompt,
            max_tokens=150,
            temperature=0.0,
        )

        if not response.success:
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "reasoning": f"Assessment failed: {response.error}",
                "success": False,
            }

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            result["success"] = True
            return result
        except json.JSONDecodeError:
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "reasoning": f"Failed to parse: {response.content[:100]}",
                "success": False,
            }


if __name__ == "__main__":
    # Test the client
    print("Testing OpenRouter client...")

    client = OpenRouterClient(model="openai/gpt-4o-mini")

    # Test basic query
    response = client.query("What is 2 + 2?")
    print(f"Response: {response.content}")
    print(f"Success: {response.success}")

    # Test explanation query
    test_prompt = "elephant bicycle quantum whisper mountain"
    explanation_query = f"What does this prompt mean? Explain in detail:\n\n\"{test_prompt}\""
    response = client.query(explanation_query)
    print(f"\nTest prompt: {test_prompt}")
    print(f"Explanation: {response.content}")
