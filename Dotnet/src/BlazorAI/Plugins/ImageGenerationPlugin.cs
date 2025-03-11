using System;
using System.ComponentModel;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.TextToImage;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Net.Http;
using System.Text.RegularExpressions;

namespace BlazorAI.Plugins
{
    public class ImageGenerationPlugin
    {
        private readonly IConfiguration _configuration;
        private ILogger<ImageGenerationPlugin> _logger;
        private readonly HttpClient _httpClient;

        public ImageGenerationPlugin(IConfiguration configuration)
        {
            _configuration = configuration;
            _httpClient = new HttpClient();
        }

        [KernelFunction("GenerateImage")]
        [Description("Generates an image based on a text description. Use this when the user wants to create, draw, or visualize an image.")]
        public async Task<string> GenerateImage(
            [Description("Detailed description of the image to generate")] string prompt,
            [Description("Size of the image (e.g., '1024x1024', '512x512')")] string size = "1024x1024",
            Kernel kernel = null)
        {
            try
            {
                // Get the logger if available
                _logger = kernel?.GetRequiredService<ILoggerFactory>()?.CreateLogger<ImageGenerationPlugin>();
                _logger?.LogInformation($"Generating image with prompt: {prompt}, size: {size}");

                // Get the text-to-image service from the kernel
                var imageService = kernel.GetRequiredService<ITextToImageService>();
                
                // Parse dimensions from the size parameter
                int width = 1024;
                int height = 1024;
                
                if (size != null && size.Contains("x"))
                {
                    var dimensions = size.Split('x');
                    if (dimensions.Length == 2 && 
                        int.TryParse(dimensions[0], out int parsedWidth) && 
                        int.TryParse(dimensions[1], out int parsedHeight))
                    {
                        width = parsedWidth;
                        height = parsedHeight;
                    }
                    else
                    {
                        _logger?.LogWarning($"Invalid size format: {size}. Using default 1024x1024.");
                    }
                }

                // Generate the image - this will return a string (either URL or base64)
                string resultString = await imageService.GenerateImageAsync(prompt, width, height, kernel);

                // Check if the result is a URL or a JSON response
                string imageUrl;
                if (Uri.IsWellFormedUriString(resultString, UriKind.Absolute))
                {
                    // It's a direct URL
                    imageUrl = resultString;
                }
                else
                {
                    // Parse the JSON response to extract the image URL
                    try
                    {
                        var jsonResponse = JsonDocument.Parse(resultString);
                        imageUrl = jsonResponse.RootElement.GetProperty("data")[0].GetProperty("url").GetString();
                    }
                    catch (Exception ex)
                    {
                        throw new InvalidOperationException($"Failed to parse image generation response: {resultString}", ex);
                    }
                }

                // Return the image URL
                return imageUrl;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, $"Error generating image: {ex.Message}");
                return $"Error generating image: {ex.Message}";
            }
        }
    }
}