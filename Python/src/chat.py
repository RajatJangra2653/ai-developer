import asyncio
import logging
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAITextToImage, AzureTextEmbedding
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.openapi_plugin import OpenAPIFunctionExecutionParameters
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
import os
from plugins.time_plugin import TimePlugin
from plugins.geo_coding_plugin import GeoPlugin
from plugins.weather_plugin import WeatherPlugin
from plugins.ContosoSearchPlugin import ContosoSearchPlugin  # Add this import
from plugins.ImageGenerationPlugin import ImageGenerationPlugin
from semantic_kernel.connectors.ai.open_ai import AzureTextToImage
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai import OpenAITextToImage  # Add this import and keep the existing OpenAITextToImage

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
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        service_id="chat-service",
    )

    # Add the chat completion service to the kernel
    kernel.add_service(chat_completion_service)

    # Add Text Embedding service for semantic search
    text_embedding_service = AzureTextEmbedding(
        deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        service_id="embedding-service"
    )
    kernel.add_service(text_embedding_service)
    logger.info("Text Embedding service added")

    # Add DALL-E image generation service
    image_generation_service = AzureTextToImage(
        deployment_name=os.getenv("AZURE_TEXT_TO_IMAGE_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_TEXT_TO_IMAGE_API_KEY"),
        endpoint=os.getenv("AZURE_TEXT_TO_IMAGE_ENDPOINT"),
        service_id="image-service"
    )
    kernel.add_service(image_generation_service)
    logger.info("DALL-E image generation service added")

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
    logger.info("GeoLocation plugin loaded")

    kernel.add_plugin(
        WeatherPlugin(),
        plugin_name="Weather",
    )
    logger.info("Weather plugin loaded")
    
    # Add Contoso Handbook Search Plugin
    kernel.add_plugin(
        ContosoSearchPlugin(),
        plugin_name="ContosoSearch",
    )
    logger.info("Contoso Handbook Search plugin loaded")
    
    # Add Image Generation Plugin
    image_plugin = ImageGenerationPlugin()
    # Set the kernel directly on the plugin instance
    image_plugin.set_kernel(kernel)
    kernel.add_plugin(
        image_plugin,
        plugin_name="ImageGeneration",
    )
    logger.info("Image Generation plugin loaded")

    kernel.add_plugin_from_openapi(
        plugin_name="get_tasks",
        openapi_document_path="http://127.0.0.1:8000/openapi.json",
        execution_settings=OpenAPIFunctionExecutionParameters(
            enable_payload_namespacing=True,
        )
    )


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

async def test_image_generation(prompt="A cute cat wearing a hat"):
    """Test function to directly generate an image"""
    kernel = initialize_kernel()
    
    # Create and register the plugin
    image_plugin = ImageGenerationPlugin()
    image_plugin.set_kernel(kernel)
    kernel.add_plugin(image_plugin, plugin_name="ImageGeneration")
    
    # Call the generate_image function directly
    try:
        logger.info(f"Testing image generation with prompt: {prompt}")
        result = await image_plugin.generate_image(prompt=prompt, kernel=kernel)
        logger.info(f"Image generation test result: {result}")
        return result
    except Exception as e:
        logger.error(f"Image generation test failed: {str(e)}")
        return f"Error: {str(e)}"

# Example usage:
# asyncio.run(test_image_generation("A beautiful mountain landscape"))