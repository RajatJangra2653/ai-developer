import os
import json
from typing import Annotated
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel import Kernel
import httpx
from PIL import Image

class ImageGenerationPlugin:
    """Plugin for generating images using DALL-E."""

    def __init__(self):
        """Initialize the ImageGenerationPlugin."""
        pass

    @kernel_function(
        description="Generates an image based on the text prompt",
        name="generate_image"
    )
    async def generate_image(
        self, 
        prompt: Annotated[str, "Text description of the image to generate"],
        size: Annotated[str, "Size of the image (default: 1024x1024)"] = "1024x1024"
    ) -> str:
        """
        Generate an image using DALL-E based on the provided text prompt.
        Returns the URL of the generated image.
        """
        try:
            kernel = self._kernel
            image_service = kernel.get_service_by_id("image-service")

            # Generate the image (with parameters similar to the reference code)
            # model and n can be adjusted as needed for your specific environment
            result = await image_service.generate_image_async(
                prompt=prompt,
                size=size,
                model="dalle3",  # Example model name
                n=1,         # Number of images to generate
                
            )

            image_dir = os.path.join(os.curdir, 'images')

            # If the directory doesn't exist, create it
            if not os.path.isdir(image_dir):
                os.mkdir(image_dir)

            # Initialize the image path (note the filetype should be png)
            image_path = os.path.join(image_dir, 'generated_image.png')

            # Retrieve the generated image
            json_response = json.loads(result.model_dump_json())
            image_url = json_response["data"][0]["url"]  # extract image URL from response
            generated_image = httpx.get(image_url).content  # download the image

            with open(image_path, "wb") as image_file:
                image_file.write(generated_image)

            # Display the image in the default image viewer
            image = Image.open(image_path)
            image.show()

            return f"Image generated successfully! URL: {image_url}"

        except Exception as e:
            return f"Error generating image: {str(e)}"