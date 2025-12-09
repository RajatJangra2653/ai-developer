import asyncio
import logging
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextToImage, AzureChatPromptExecutionSettings
from semantic_kernel.connectors.azure_ai_search import AzureAISearchCollection
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.openapi_plugin import OpenAPIFunctionExecutionParameters
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import KernelArguments
import os
from pathlib import Path

from plugins.ai_search_plugin import AiSearchPlugin
from plugins.geo_coding_plugin import GeoPlugin
# Challenge 03 - Import plugins you create
# from plugins.time_plugin import TimePlugin
# from plugins.weather_plugin import WeatherPlugin
# Challenge 07 - Import image plugin
# from plugins.image_plugin import ImagePlugin

# Add Logger
logger = logging.getLogger(__name__)

load_dotenv(override=True)

chat_history = ChatHistory()

def initialize_kernel():
   #Challene 02 - Add Kernel
   kernel = Kernel()
   #Challenge 02 - Chat Completion Service
   #Challenge 05 - Add Text Embedding service for semantic search
   #Challenge 07 - Add DALL-E image generation service
   return kernel


async def process_message(user_input):
    """Legacy function - delegates to ChatService"""
    chat_service = get_chat_service()
    return await chat_service.process_message(user_input)

def reset_chat_history():
    global chat_history
    chat_history = ChatHistory()
