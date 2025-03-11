using Microsoft.AspNetCore.Components;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Agents.Chat;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System.Threading;
using Microsoft.Extensions.Logging;
using System;

#pragma warning disable SKEXP0110 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

namespace BlazorAI.Components.Pages
{
    public partial class MultiAgent
    {
        private ChatHistory? chatHistory;
        private IChatCompletionService? chatCompletionService;
        private OpenAIPromptExecutionSettings? openAIPromptExecutionSettings;
        private Kernel? kernel;

        // Keep only one declaration of MessageInput and loading

        [Inject]
        public required IConfiguration Configuration { get; set; }
        
        [Inject]
        private ILoggerFactory LoggerFactory { get; set; } = null!;
        
        [Inject]
        private ILogger<MultiAgent> Logger { get; set; }

        private List<ChatCompletionAgent> Agents { get; set; } = [];
        private AgentGroupChat AgentGroupChat;

        protected async Task InitializeSemanticKernel()
        {
            chatHistory = [];

            var kernelBuilder = Kernel.CreateBuilder();

            kernelBuilder.AddAzureOpenAIChatCompletion(
                Configuration["AOI_DEPLOYMODEL"] ?? "gpt-35-turbo",
                Configuration["AOI_ENDPOINT"]!,
                Configuration["AOI_API_KEY"]!);

            kernelBuilder.Services.AddSingleton(LoggerFactory);

            kernel = kernelBuilder.Build();

            await CreateAgents();
        }

        private async Task CreateAgents()
        {
            string BusinessAnalystInstructions = """
                You are a Business Analyst which will take the requirements from the user (also known as a 'customer') and create a project plan for creating the requested app. The Business Analyst understands the user requirements and creates detailed documents with requirements and costing. The documents should be usable by the SoftwareEngineer as a reference for implementing the required features, and by the Product Owner for reference to determine if the application delivered by the Software Engineer meets all of the user's requirements.
            """;

            string SoftwareEngineerInstructions = """
                You are a Software Engineer, and your goal is create a web app using HTML and JavaScript by taking into consideration all the requirements given by the Business Analyst. The application should implement all the requested features. Deliver the code to the Product Owner for review when completed. You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear.
            """;

            string ProductOwnerInstructions = """
                You are the Product Owner which will review the software engineer's code to ensure all user requirements are completed. You are the guardian of quality, ensuring the final product meets all specifications and receives the green light for release. Once all client requirements are completed, you can approve the request by just responding "%APPR%". Do not ask any other agent or the user for approval. If there are missing features, you will need to send a request back to the SoftwareEngineer or BusinessAnalyst with details of the defect. To approve, respond with the token %APPR%.
            """;

            #pragma warning disable SKEXP0110, SKEXP0001 // Rethrow to preserve stack details

            // Create a Business Analyst Agent
            ChatCompletionAgent BusinessAnalystAgent = new()
            {
                Instructions = BusinessAnalystInstructions,
                Name = "BusinessAnalyst",
                Kernel = kernel
            };
            Agents.Add(BusinessAnalystAgent);

            // Create a Software Engineer Agent
            ChatCompletionAgent SoftwareEngineerAgent = new()
            {
                Instructions = SoftwareEngineerInstructions,
                Name = "SoftwareEngineer",
                Kernel = kernel
            };
            Agents.Add(SoftwareEngineerAgent);

            // Create a Product Owner Agent
            ChatCompletionAgent ProductOwnerAgent = new()
            {
                Instructions = ProductOwnerInstructions,
                Name = "ProductOwner",
                Kernel = kernel
            };
            Agents.Add(ProductOwnerAgent);
            
            // Setup the group chat with approval termination
            var productOwnerAgentRef = Agents.FirstOrDefault(a => a.Name == "ProductOwner");
            
            // Create the group chat
            AgentGroupChat = new AgentGroupChat(
                BusinessAnalystAgent, SoftwareEngineerAgent, ProductOwnerAgent)
            {
                ExecutionSettings = new()
                {
                    TerminationStrategy = new ApprovalTerminationStrategy()
                    {
                        Agents = [ProductOwnerAgent],
                        MaximumIterations = 25,
                    }
                }
            };
        }

        private async Task SendMessage()
        {
            if (string.IsNullOrWhiteSpace(MessageInput) || chatHistory == null || kernel == null || AgentGroupChat == null)
                return;

            // Copy the message from the user input
            var userMessage = MessageInput;
            MessageInput = string.Empty;
            loading = true;
            chatHistory.AddUserMessage(userMessage);
            StateHasChanged();

            try
            {
                // Create chat message content
                var chatMessageContent = new ChatMessageContent(
                    AuthorRole.User,
                    userMessage
                );

                // Use async foreach to process messages from the agents
                await foreach (var message in AgentGroupChat.InvokeAsync(CancellationToken.None))
                {
                    if (message.Role == AuthorRole.Assistant)
                    {
                        string formattedMessage = $"{message.AuthorName}: {message.Content}";
                        chatHistory.AddAssistantMessage(formattedMessage);
                    }
                    else if (message.Role == AuthorRole.User)
                    {
                        chatHistory.AddUserMessage(message.Content);
                    }
                    else if (message.Role == AuthorRole.System)
                    {
                        // Handle system messages if needed
                    }
                    
                    // Update UI after each message
                    StateHasChanged();
                }
            }
            catch (Exception ex)
            {
                chatHistory.AddAssistantMessage($"Error: {ex.Message}");
                Logger.LogError(ex, "Error while trying to send message to agents.");
            }
            finally
            {
                loading = false;
                StateHasChanged();
            }
        }
    }

    // Fix the nested class issue by removing the nested ApprovalTerminationStrategy class
    sealed class ApprovalTerminationStrategy : TerminationStrategy
    {
        public IReadOnlyList<Agent> Agents { get; set; } = Array.Empty<Agent>();
        public int MaximumIterations { get; set; } = 25;
        private int _currentIteration = 0;
        
        protected override Task<bool> ShouldAgentTerminateAsync(Agent agent, IReadOnlyList<ChatMessageContent> history, CancellationToken cancellationToken = default)
        {
            // Check if we've exceeded the maximum number of iterations
            if (++_currentIteration > MaximumIterations)
            {
                return Task.FromResult(true);
            }

            // No history to check
            if (history.Count == 0)
            {
                return Task.FromResult(false);
            }

            // Get the last message
            var lastMessage = history[^1];

            // Check if message contains the approval token
            bool containsApprovalToken = lastMessage.Content?.Contains("%APPR%", StringComparison.OrdinalIgnoreCase) ?? false;

            // If there are specific approver agents defined, check if the message is from one of them
            if (containsApprovalToken && Agents.Count > 0)
            {
                // Only terminate if the message with the approval token is from an authorized approver
                return Task.FromResult(containsApprovalToken && Agents.Any(a => a.Name == lastMessage.AuthorName));
            }

            // If no specific approvers defined, any agent with the token can terminate
            return Task.FromResult(containsApprovalToken);
        }
    }
}
