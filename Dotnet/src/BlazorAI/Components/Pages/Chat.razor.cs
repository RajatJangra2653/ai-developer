using Microsoft.AspNetCore.Components;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using BlazorAI.Plugins;
using System;

#pragma warning disable SKEXP0040 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
#pragma warning disable SKEXP0020 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
#pragma warning disable SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

namespace BlazorAI.Components.Pages;

public partial class Chat
{
    private ChatHistory? chatHistory;
    private Kernel? kernel;
    private OpenAIPromptExecutionSettings? promptSettings;

    [Inject]
    public required IConfiguration Configuration { get; set; }
    [Inject]
    private ILoggerFactory LoggerFactory { get; set; } = null!;

    protected async Task InitializeSemanticKernel()
    {
        // Initialize chat history with a system message about available functions
        chatHistory = new ChatHistory();
        // Challenge 02 - Configure Semantic Kernel
        var kernelBuilder = Kernel.CreateBuilder();

        // Challenge 02 - Add OpenAI Chat Completion
        kernelBuilder.AddAzureOpenAIChatCompletion(
            Configuration["AOI_DEPLOYMODEL"]!,
            Configuration["AOI_ENDPOINT"]!,
            Configuration["AOI_API_KEY"]!);

        // Add Logger for Kernel
        kernelBuilder.Services.AddSingleton(LoggerFactory);

        // Challenge 03 and 04 - Services Required
        kernelBuilder.Services.AddHttpClient();

        // Challenge 05 - Register Azure AI Foundry Text Embeddings Generation

        // Challenge 05 - Register Search Index

        // Challenge 07 - Add Azure AI Foundry Text To Image

        // Challenge 02 - Finalize Kernel Builder
        kernel = kernelBuilder.Build();

        // Challenge 03, 04, 05, & 07 - Add Plugins
        await AddPlugins();

        // Challenge 03 - Create OpenAIPromptExecutionSettings for automatic function calling
        promptSettings = new OpenAIPromptExecutionSettings
        {
            // Use this setting to let the model automatically invoke functions
            ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions,
            Temperature = 0.7,
            TopP = 0.95,
            MaxTokens = 800
        };
    }

    private async Task AddPlugins()
    {
        // Challenge 03 - Add Time Plugin
        var timePlugin = new BlazorAI.Plugins.TimePlugin();
        kernel.ImportPluginFromObject(timePlugin, "TimePlugin");

        // Challenge 04 - Import OpenAPI Spec

        // Challenge 05 - Add Search Plugin

        // Challenge 07 - Text To Image Plugin

    }

    private async Task SendMessage()
    {
        if (!string.IsNullOrWhiteSpace(newMessage) && chatHistory != null)
        {
            // This tells Blazor the UI is going to be updated.
            StateHasChanged();
            loading = true;
            // Copy the user message to a local variable and clear the newMessage field in the UI
            var userMessage = newMessage;
            newMessage = string.Empty;
            StateHasChanged();

            // Add user message to chat history
            chatHistory.AddUserMessage(userMessage);

            // Get chat completion service from kernel
            var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

            // Send the message to get assistant response with explicit kernel parameter
            var assistantResponse = await chatCompletionService.GetChatMessageContentAsync(
                chatHistory: chatHistory,
                executionSettings: promptSettings,
                kernel: kernel);  // Pass the kernel explicitly

            // Add the assistant's response to the chat history
            chatHistory.AddAssistantMessage(assistantResponse.Content);

            loading = false;
        }
    }
}
