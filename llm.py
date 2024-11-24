from langchain_openai import ChatOpenAI
from typing import Optional
from pydantic import BaseModel, Field
import os
from enum import IntEnum

class CapabilityLevel(IntEnum):
    BASIC = 0
    LOW = 1
    MODERATE = 2
    STANDARD = 3
    HIGH = 4
    ADVANCED = 5

class LLMConfig(BaseModel):
    api_base: str = Field(..., description="Base URL for the API endpoint")
    api_key: str = Field(..., description="API key for authentication")
    model_name: str = Field(..., description="Name of the model to use")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Temperature for response generation")

def get_llm_config(capability_level: CapabilityLevel) -> LLMConfig:
    """
    Get the LLM configuration based on capability level.

    Args:
        capability_level (CapabilityLevel): The desired capability level

    Returns:
        LLMConfig: Configuration for the LLM
    """
    base_config = {
        "api_base": os.environ.get("OPENAI_API_BASE", ""),
        "api_key": os.environ.get("OPENAI_API_KEY", "")
    }

    # model_configs = {
    #     CapabilityLevel.ADVANCED: ("o1-preview", 1),
    #     CapabilityLevel.HIGH: ("o1-mini", 1),
    #     CapabilityLevel.STANDARD: ("gpt-4o", 0.7),
    #     CapabilityLevel.MODERATE: ("gpt-4", 0.7),
    #     CapabilityLevel.LOW: ("gpt-4o-mini", 0.7),
    #     CapabilityLevel.BASIC: ("gpt-3.5-turbo", 0.7)
    # }

    model_configs = {
        CapabilityLevel.ADVANCED: ("gpt-3.5-turbo", 0.7),
        CapabilityLevel.HIGH: ("gpt-3.5-turbo", 0.7),
        CapabilityLevel.STANDARD: ("gpt-3.5-turbo", 0.7),
        CapabilityLevel.MODERATE: ("gpt-3.5-turbo", 0.7),
        CapabilityLevel.LOW: ("gpt-3.5-turbo", 0.7),
        CapabilityLevel.BASIC: ("gpt-3.5-turbo", 0.7)
    }

    model_name, temperature = model_configs[capability_level]
    return LLMConfig(
        api_base=base_config["api_base"],
        api_key=base_config["api_key"],
        model_name=model_name,
        temperature=temperature
    )

def get_llm(capability_level: int) -> ChatOpenAI:
    """
    Returns an LLM instance configured to use a custom OpenAI-style endpoint.

    Args:
        capability_level (int): Integer representing the desired capability level (0-5)

    Returns:
        ChatOpenAI: Configured LLM instance

    Raises:
        ValueError: If capability_level is not in range 0-5
    """
    try:
        level = CapabilityLevel(capability_level)
    except ValueError:
        raise ValueError(f"Capability level must be between {CapabilityLevel.BASIC} and {CapabilityLevel.ADVANCED}")

    config = get_llm_config(level)

    return ChatOpenAI(
        openai_api_base=config.api_base,
        openai_api_key=config.api_key,
        model_name=config.model_name,
        temperature=config.temperature
    )
