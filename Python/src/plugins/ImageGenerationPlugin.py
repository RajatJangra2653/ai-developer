import os
import json
from typing import Annotated
from semantic_kernel.functions import kernel_function, KernelFunction
from semantic_kernel.kernel import Kernel
import httpx
from PIL import Image

class ImageGenerationPlugin:
    """Plugin for generating images using DALL-E."""

    def __init__(self):
        """Initialize the ImageGenerationPlugin."""
        self._kernel = None

    # This method will be called by Semantic Kernel when the plugin is registered
    def set_kernel(self, kernel):
        self._kernel = kernel

    @kernel_function(
        description="Generates an image based on the text prompt",
        name="generate_image"
    )
    async def generate_image(
        self, 
        prompt: Annotated[str, "Text description of the image to generate"],
        size: Annotated[str, "Size of the image (default: 1024x1024)"] = "1024x1024",
        kernel=None  # Allow kernel to be passed as a parameter
    ) -> str:
        """
        Generate an image using DALL-E based on the provided text prompt.
        Returns the URL of the generated image.
        """
        try:
            # Use the provided kernel or the stored one
            kernel_to_use = kernel or self._kernel
            if not kernel_to_use:
                return "Error: No kernel available to the plugin"
                
            # Get the image service - use correct method name
            try:
                image_service = kernel_to_use.get_service(service_id="image-service")
            except Exception as e:
                return f"Error accessing image service: {str(e)}"

            print(f"Generating image with prompt: {prompt}")
            
            # Parse size (format like "1024x1024")
            if "x" in size:
                width, height = map(int, size.split('x'))
            else:
                # Default to square if size format is incorrect
                width = height = 1024
            
            # Generate the image with correct parameter names
            result = await image_service.generate_image(
                description=prompt,  # Using prompt as the description
                width=width,
                height=height
            )

            image_dir = os.path.join(os.curdir, 'images')

            # If the directory doesn't exist, create it
            if not os.path.isdir(image_dir):
                os.mkdir(image_dir)

            # Initialize the image path
            image_path = os.path.join(image_dir, 'generated_image.png')

            # Properly handle the result based on its type
            try:
                # For newer SDK versions that return a string
                if isinstance(result, str):
                    print(f"Result is a string: {result}")
                    json_response = json.loads(result)
                # For older SDK versions that return an object with model_dump_json
                else:
                    print(f"Result is an object with type: {type(result)}")
                    json_response = json.loads(result.model_dump_json())
                
                print(f"API Response: {json_response}")
                
                image_url = json_response["data"][0]["url"]  # extract image URL from response
                generated_image = httpx.get(image_url).content  # download the image

                with open(image_path, "wb") as image_file:
                    image_file.write(generated_image)

                # Display the image in the default image viewer
                image = Image.open(image_path)
                image.show()

                return f"Image generated successfully! URL: {image_url}"
            except Exception as e:
                # If we can't parse the response properly, log it and return it as-is
                print(f"Error processing image response: {str(e)}")
                print(f"Raw response: {result}")
                
                # Try a direct approach if the result is already the URL
                if result.startswith("http"):
                    try:
                        generated_image = httpx.get(result).content
                        with open(image_path, "wb") as image_file:
                            image_file.write(generated_image)
                        image = Image.open(image_path)
                        image.show()
                        return f"Image generated successfully! URL: {result}"
                    except:
                        pass
                        
                return f"Image response received but couldn't process it: {result}"

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error generating image: {str(e)}\n{error_details}")
            return f"Error generating image: {str(e)}"