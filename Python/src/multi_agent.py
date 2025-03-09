import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import KernelFunctionTerminationStrategy
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import KernelFunctionSelectionStrategy
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions.kernel_function_metadata import KernelFunctionMetadata
from semantic_kernel.functions.kernel_function import KernelFunction


async def run_multi_agent(input: str):
    """Implement the multi-agent system."""
    
    load_dotenv()
    kernel = Kernel()
    
    chat_completion_service = AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        service_id="chat-service",
    )
    kernel.add_service(chat_completion_service)
    
    # Define agent personas
    business_analyst_persona = """You are a Business Analyst which will take the requirements from the user (also known as a 'customer')
and create a project plan for creating the requested app. The Business Analyst understands the user
requirements and creates detailed documents with requirements and costing. The documents should be 
usable by the SoftwareEngineer as a reference for implementing the required features, and by the 
Product Owner for reference to determine if the application delivered by the Software Engineer meets
all of the user's requirements."""
    
    software_engineer_persona = """You are a Software Engineer, and your goal is create a web app using HTML and JavaScript
by taking into consideration all the requirements given by the Business Analyst. The application should
implement all the requested features. Deliver the code to the Product Owner for review when completed.
You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear."""

    product_owner_persona = """You are the Product Owner which will review the software engineer's code to ensure all user 
requirements are completed. You are the guardian of quality, ensuring the final product meets
all specifications and receives the green light for release. Once all client requirements are
completed, you can approve the request by just responding "%APPR%". Do not ask any other agent
or the user for approval. If there are missing features, you will need to send a request back
to the SoftwareEngineer or BusinessAnalyst with details of the defect. To approve, respond with
the token %APPR%."""
    
    # Create agents
    business_analyst = ChatCompletionAgent(
        name="BusinessAnalyst",
        kernel=kernel,
    )
    software_engineer = ChatCompletionAgent(
        name="SoftwareEngineer",
        kernel=kernel,
    )
    product_owner = ChatCompletionAgent(
        name="ProductOwner",
        kernel=kernel,
    )
    
    # Create functions using KernelFunction directly
    # Create termination function
    termination_prompt = """
    Check the conversation history for a message by the ProductOwner containing "%APPR%". 
    If found, respond with 'yes' to indicate termination. Otherwise, respond with 'no'.

    History:
    {{$history}}
    """
    
    # Create a termination function
    termination_function = KernelFunction.from_prompt(
        plugin_name="ConversationManager",  # Add this required parameter
        prompt=termination_prompt,
        function_name="termination",
        description="Determines if conversation should terminate based on approval"
    )
    kernel.add_function(termination_function)
    
    # Create selection function
    selection_prompt = """
    Based on the conversation history, determine which agent should speak next.
    Consider the following:
    - If this is the start of the conversation, the {BusinessAnalyst} should speak first to understand requirements
    - If the BusinessAnalyst has outlined requirements, the {SoftwareEngineer} should implement them
    - If the {SoftwareEngineer} has presented code, the {ProductOwner} should review it
    - If the {ProductOwner} has requested changes, the {SoftwareEngineer} should address them
    - If there are requirement questions, the {BusinessAnalyst} should clarify
    
    Return only the name of the agent who should speak next: BusinessAnalyst, SoftwareEngineer, or ProductOwner.
    
    History:
    {{$history}}
    
    Last agent to speak: {{$last_agent}}
    """
    
    selection_function = KernelFunction.from_prompt(
        plugin_name="ConversationManager",  # Add this required parameter
        prompt=selection_prompt,
        function_name="selection", 
        description="Determines which agent should speak next in the conversation"
    )
    kernel.add_function(selection_function)

    # Create a helper function for creating a kernel with chat completion
    def _create_kernel_with_chat_completion(function_name):
        """Create a kernel with chat completion service for the specified function."""
        temp_kernel = Kernel()
        temp_kernel.add_service(AzureChatCompletion(
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            service_id="chat-service",
        ))
        return temp_kernel

    # Create agent group chat with both termination and selection functions
    group_chat = AgentGroupChat(
        agents=[business_analyst, software_engineer, product_owner],
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[product_owner],  # Only the ProductOwner can approve and terminate
            kernel=kernel,  # Use the main kernel instead of creating a new one
            function=termination_function,  # Use the termination function we created
            result_parser=lambda result: str(result.value[0]).lower() == "yes",
            history_variable_name="history",
            maximum_iterations=20
        ),
        next_agent_selection_strategy=KernelFunctionSelectionStrategy(
            kernel=kernel,
            function=selection_function,  # Use the selection function we created
            history_variable_name="history",
        )
    )
    
    # Inject personas as system messages
    messages = [
        ChatMessageContent(role=AuthorRole.SYSTEM, content=business_analyst_persona, author="BusinessAnalyst"),
        ChatMessageContent(role=AuthorRole.SYSTEM, content=software_engineer_persona, author="SoftwareEngineer"),
        ChatMessageContent(role=AuthorRole.SYSTEM, content=product_owner_persona, author="ProductOwner"),
        ChatMessageContent(role=AuthorRole.USER, content=input, author="User"),
    ]
    
    responses = []
    print("Starting multi-agent conversation...")
    
    async for msg in group_chat.invoke_async(messages):
        if isinstance(msg, ChatMessageContent):
            print(f"# {msg.role} - {msg.name or msg.author or '*'}: '{msg.content}'")
            responses.append(msg)
    
    print("Multi-agent conversation completed.")
    return responses
