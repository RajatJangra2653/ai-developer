using System.ComponentModel;
using System.Text.Json.Serialization;
using Azure;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Models;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using System.Text;

namespace BlazorAI.Plugins
{
    public class ContosoSearchPlugin
    {
        private readonly ITextEmbeddingGenerationService _textEmbeddingGenerationService;
        private readonly SearchIndexClient _indexClient;

        public ContosoSearchPlugin(IConfiguration configuration)
        {
            // Create the search index client
            _indexClient = new SearchIndexClient(
                new Uri(configuration["AI_SEARCH_URL"]),
                new AzureKeyCredential(configuration["AI_SEARCH_KEY"]));

            // Get the embedding service from the kernel
            var kernelBuilder = Kernel.CreateBuilder();
            kernelBuilder.AddAzureOpenAITextEmbeddingGeneration(
                configuration["EMBEDDINGS_DEPLOYMODEL"],
                configuration["AOI_ENDPOINT"],
                configuration["AOI_API_KEY"]);
            var kernel = kernelBuilder.Build();
            _textEmbeddingGenerationService = kernel.GetRequiredService<ITextEmbeddingGenerationService>();
        }

        [KernelFunction("SearchHandbook")]
        [Description("Searches the Contoso employee handbook for information about company policies, benefits, procedures or other employee-related questions. Use this when the user asks about company policies, employee benefits, work procedures, or any information that might be in an employee handbook.")]
        public async Task<string> Search(
            [Description("The user's question about company policies, benefits, procedures or other handbook-related information")] string query)
        {
            try
            {
                // Convert string query to vector embedding
                ReadOnlyMemory<float> embedding = await _textEmbeddingGenerationService.GenerateEmbeddingAsync(query);

                // Get client for search operations
                SearchClient searchClient = _indexClient.GetSearchClient("employeehandbook");

                // Configure request parameters
                VectorizedQuery vectorQuery = new(embedding);
                vectorQuery.Fields.Add("contentVector");  // The vector field in your index
                vectorQuery.KNearestNeighborsCount = 3;   // Get top 3 matches

                SearchOptions searchOptions = new()
                {
                    VectorSearch = new() { Queries = { vectorQuery } },
                    Size = 3  // Return top 3 results
                };

                // Perform search request
                Response<SearchResults<IndexSchema>> response = await searchClient.SearchAsync<IndexSchema>(searchOptions);

                // Collect search results
                StringBuilder results = new StringBuilder();
                await foreach (SearchResult<IndexSchema> result in response.Value.GetResultsAsync())
                {
                    if (!string.IsNullOrEmpty(result.Document.Content))
                    {
                        results.AppendLine($"Title: {result.Document.Title}");
                        results.AppendLine($"Content: {result.Document.Content}");
                        results.AppendLine();
                    }
                }

                return results.Length > 0 
                    ? results.ToString()
                    : "No relevant information found in the employee handbook.";
            }
            catch (Exception ex)
            {
                return $"Search error: {ex.Message}";
            }
        }

        private sealed class IndexSchema
        {
            [JsonPropertyName("content")]
            public string Content { get; set; }

            [JsonPropertyName("title")]
            public string Title { get; set; }

            [JsonPropertyName("url")]
            public string Url { get; set; }
        }
    }
}