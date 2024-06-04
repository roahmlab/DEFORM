from PIL import Image


def increase_resolution(image_path, output_path, scale_factor):
    # Open an image file
    with Image.open(image_path) as img:
        # Calculate the new size
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))

        # Resize the image
        img_resized = img.resize(new_size, Image.LANCZOS)  # LANCZOS is good for upsampling

        # Save the resized image
        img_resized.save(output_path)
        print(f"Image saved with increased resolution: {output_path}")


# Usage
increase_resolution("contribution.png", "contribution2.png", 4)  # Increase resolution by factor of 2
