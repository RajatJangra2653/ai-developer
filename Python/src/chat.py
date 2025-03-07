import asyncio
import logging
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAITextToImage
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.openapi_plugin import OpenAPIFunctionExecutionParameters
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
import os
from plugins.time_plugin import TimePlugin
from plugins.geo_coding_plugin import GeoPlugin  # Add this import statement
from plugins.weather_plugin import WeatherPlugin  # Add this import at the top of the file
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Add Logger
logger = logging.getLogger(__name__)

load_dotenv(override=True)

chat_history = ChatHistory()

def initialize_kernel():
    # Challenge 02 - Add Kernel
    kernel = Kernel()

    # Challenge 02 - Chat Completion Service
    chat_completion_service = AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # Used to point to your service
        service_id="chat-service",
    )

    # Add the chat completion service to the kernel
    kernel.add_service(chat_completion_service)

    # Retrieve the chat completion service by type
    chat_completion_service = kernel.get_service(type=ChatCompletionClientBase)

    # Retrieve the default inference settings
    execution_settings = kernel.get_prompt_execution_settings_from_service_id("chat-service")

    # Challenge 02 - Add kernel to the chat completion service
    return kernel


async def process_message(user_input):
    kernel = initialize_kernel()

    # Challenge 03 - Create Prompt Execution Settings
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    logger.info("Automatic function calling enabled")

    # Challenge 03 - Add Time Plugin
    time_plugin = TimePlugin()
    kernel.add_plugin(time_plugin, plugin_name="TimePlugin")
    logger.info("Time plugin loaded")

    kernel.add_plugin(
          GeoPlugin(),
          plugin_name="GeoLocation",
    )

    kernel.add_plugin(
        WeatherPlugin(),
        plugin_name="Weather",
    )
    logger.info("Weather plugin loaded")

    # Add user input to chat history
    global chat_history
    chat_history.add_user_message(user_input)

    # Get the chat completion service
    chat_completion = kernel.get_service(type=ChatCompletionClientBase)
    
    # Make sure to pass the execution_settings with AUTO function calling
    # and pass kernel to allow access to the functions
    response = await chat_completion.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel  # Pass the kernel with the registered plugin
    )

    # Add the AI's response to the chat history
    chat_history.add_assistant_message(str(response))
    
    logger.info(f"Response: {response}")
    return response

def reset_chat_history():
    global chat_history
    chat_history = ChatHistory()
