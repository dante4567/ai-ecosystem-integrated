#!/usr/bin/env python3

import os
from typing import Dict, Any, Optional
from enum import Enum

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    GOOGLE = "google"

class LLMConfig:
    """Shared LLM configuration for all services"""

    def __init__(self):
        self.providers = {
            LLMProvider.OPENAI: {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": "https://api.openai.com/v1",
                "models": {
                    "chat": "gpt-4-turbo-preview",
                    "embedding": "text-embedding-3-small",
                    "completion": "gpt-3.5-turbo-instruct"
                },
                "max_tokens": 4096,
                "temperature": 0.7
            },
            LLMProvider.ANTHROPIC: {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "base_url": "https://api.anthropic.com",
                "models": {
                    "chat": "claude-3-sonnet-20240229",
                    "completion": "claude-3-haiku-20240307"
                },
                "max_tokens": 4096,
                "temperature": 0.7
            },
            LLMProvider.GROQ: {
                "api_key": os.getenv("GROQ_API_KEY"),
                "base_url": "https://api.groq.com/openai/v1",
                "models": {
                    "chat": "mixtral-8x7b-32768",
                    "fast_chat": "llama2-70b-4096"
                },
                "max_tokens": 32768,
                "temperature": 0.7
            },
            LLMProvider.GOOGLE: {
                "api_key": os.getenv("GOOGLE_API_KEY"),
                "base_url": "https://generativelanguage.googleapis.com/v1",
                "models": {
                    "chat": "gemini-pro",
                    "vision": "gemini-pro-vision"
                },
                "max_tokens": 8192,
                "temperature": 0.7
            }
        }

        # Default provider selection
        self.default_provider = LLMProvider.OPENAI
        self.fallback_providers = [LLMProvider.GROQ, LLMProvider.ANTHROPIC]

    def get_provider_config(self, provider: LLMProvider) -> Dict[str, Any]:
        """Get configuration for a specific provider"""
        return self.providers.get(provider, {})

    def get_api_key(self, provider: LLMProvider) -> Optional[str]:
        """Get API key for a specific provider"""
        config = self.get_provider_config(provider)
        return config.get("api_key")

    def get_model(self, provider: LLMProvider, model_type: str = "chat") -> Optional[str]:
        """Get model name for a specific provider and type"""
        config = self.get_provider_config(provider)
        models = config.get("models", {})
        return models.get(model_type)

    def is_provider_available(self, provider: LLMProvider) -> bool:
        """Check if a provider is configured and available"""
        api_key = self.get_api_key(provider)
        return api_key is not None and api_key.strip() != ""

    def get_available_providers(self) -> list[LLMProvider]:
        """Get list of available providers"""
        return [provider for provider in LLMProvider if self.is_provider_available(provider)]

    def get_best_provider(self, task_type: str = "general") -> LLMProvider:
        """Get the best provider for a specific task type"""
        # Task-specific provider selection
        task_preferences = {
            "fast": [LLMProvider.GROQ, LLMProvider.OPENAI],
            "reasoning": [LLMProvider.ANTHROPIC, LLMProvider.OPENAI],
            "coding": [LLMProvider.OPENAI, LLMProvider.ANTHROPIC],
            "vision": [LLMProvider.GOOGLE, LLMProvider.OPENAI],
            "general": [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GROQ]
        }

        preferred_providers = task_preferences.get(task_type, task_preferences["general"])

        for provider in preferred_providers:
            if self.is_provider_available(provider):
                return provider

        # Fallback to any available provider
        available = self.get_available_providers()
        return available[0] if available else self.default_provider

    def get_request_config(self, provider: LLMProvider, **overrides) -> Dict[str, Any]:
        """Get request configuration for API calls"""
        config = self.get_provider_config(provider).copy()

        # Apply overrides
        for key, value in overrides.items():
            config[key] = value

        return {
            "api_key": config["api_key"],
            "base_url": config["base_url"],
            "model": config["models"].get("chat"),
            "max_tokens": config.get("max_tokens", 4096),
            "temperature": config.get("temperature", 0.7)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            "providers": {provider.value: config for provider, config in self.providers.items()},
            "default_provider": self.default_provider.value,
            "available_providers": [p.value for p in self.get_available_providers()]
        }


# Global configuration instance
llm_config = LLMConfig()


def get_llm_config() -> LLMConfig:
    """Get the global LLM configuration instance"""
    return llm_config