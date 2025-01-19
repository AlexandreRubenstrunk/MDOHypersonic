from PIL import Image
import os


# Define whether to use propulsion module images or aerodynamic module images
# Set to True to use propulsion images, False otherwise
Propu = False  
# Set to True to use aerodynamic images, False otherwise
Aero = True  

# Get the current working directory and truncate it to the MDOHypersonic root folder
Actual_Path = os.getcwd()
Actual_Path = Actual_Path.split("MDOHypersonic")[0]

# Determine the path to the images folder based on the chosen module
if Propu:
    chemin_images = str(Actual_Path + "MDOHypersonic\\Module_Propu\\ImageGIf")
elif Aero:
    chemin_images = str(Actual_Path + "MDOHypersonic\\Module_Aero\\ImageGIf")

# Retrieve all files with a .png extension, sort them numerically by filename
fichiers_images = sorted(
    [f for f in os.listdir(chemin_images) if f.endswith(".png")],
    key=lambda x: int(os.path.splitext(x)[0])  # Sort by the numeric value in the filename
)

# Load the images into a list using the PIL library
images = [Image.open(os.path.join(chemin_images, f)) for f in fichiers_images]

# Define the output path for the GIF
chemin_gif = str(Actual_Path + "MDOHypersonic\\MDO\\CreatGif\\Optimisation.gif")

# Create the GIF using the loaded images
images[0].save(
    chemin_gif,
    save_all=True,
    append_images=images[1:],  # Add the remaining images to the GIF
    duration=100,             # Display duration for each image in milliseconds
    loop=0                    # Number of loops: 0 for infinite looping
)

# Print the path of the created GIF
print(f"GIF created: {chemin_gif}")

