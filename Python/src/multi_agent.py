import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.agents.strategies.termination.default_termination_strategy import DefaultTerminationStrategy
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel import Kernel


class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""
 
    def __init__(self, maximum_iterations=20):
        """Initialize with a maximum number of iterations as a safety mechanism."""
        self.maximum_iterations = maximum_iterations
        self.current_iterations = 0
 
    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        # Increment iteration counter
        self.current_iterations += 1
        
        # Check for maximum iterations as a safety measure
        if self.current_iterations >= self.maximum_iterations:
            print("Maximum iterations reached. Terminating conversation.")
            return True
            
        # Check if history exists and has messages
        if not history or len(history) == 0:
            return False
            
        # Get the last message in the history
        last_message = history[-1]
        
        # Check if the last message contains the approval token
        if isinstance(last_message, ChatMessageContent) and "%APPR%" in last_message.content:
            return True
            
        return False


async def run_multi_agent(input: str):
    """Implement the multi-agent system."""
    # Load environment variables
    load_dotenv()
    
    # Create Kernel
    kernel = Kernel()
    
    # Add chat completion service
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
        persona=business_analyst_persona,
        auto_function_calling=FunctionChoiceBehavior.Auto(),
    )
    
    software_engineer = ChatCompletionAgent(
        name="SoftwareEngineer",
        kernel=kernel,
        persona=software_engineer_persona,
        auto_function_calling=FunctionChoiceBehavior.Auto(),
    )
    
    product_owner = ChatCompletionAgent(
        name="ProductOwner",
        kernel=kernel,
        persona=product_owner_persona,
        auto_function_calling=FunctionChoiceBehavior.Auto(),
    )
    
    # Create agent group chat with termination strategy
    group_chat = AgentGroupChat(
        agents=[business_analyst, software_engineer, product_owner],
        termination_strategy=ApprovalTerminationStrategy(maximum_iterations=20),
        next_agent_selection_strategy=KernelFunctionSelectionStrategy(kernel=kernel),
    )
    
    # Set initial messages for the chat
    messages = [ChatMessageContent(
        role=AuthorRole.USER,
        content=input,
        author="User"
    )]
    
    # Create a list to collect all responses
    responses = []
    
    # Execute the conversation using async iterator with simpler pattern
    print("Starting multi-agent conversation...")
    
    # Start the conversation
    async for msg in group_chat.invoke_async(messages):
        if isinstance(msg, ChatMessageContent):
            print(f"# {msg.role} - {msg.name or msg.author or '*'}: '{msg.content}'")
            responses.append(msg)
    
    print("Multi-agent conversation completed.")
    return responses
