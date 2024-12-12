from PIL import Image
import os


Propu = True
Aero=False

Actual_Path = os.getcwd()
Actual_Path = Actual_Path.split("MDOHypersonic")[0]
if Propu == True:
    chemin_images = str(Actual_Path + "MDOHypersonic\\Module_Propu\\ImageGIf")
elif Aero == True:
    chemin_images = str(Actual_Path + "MDOHypersonic\\Module_Aero\\ImageGIf")

# Récupérer les fichiers nommés numériquement et les trier
fichiers_images = sorted(
    [f for f in os.listdir(chemin_images) if f.endswith(".png")],
    key=lambda x: int(os.path.splitext(x)[0])  # Trier par numéro
)

# Charger les images
images = [Image.open(os.path.join(chemin_images, f)) for f in fichiers_images]

# Créer le GIF
chemin_gif = str(Actual_Path + "MDOHypersonic\\MDO\\CreatGif\\Optimisation.gif")
images[0].save(
    chemin_gif,
    save_all=True,
    append_images=images[1:],  # Ajouter les autres images
    duration=100,             # Durée de chaque image en millisecondes
    loop=0                    # Nombre de boucles : 0 pour un GIF infini
)

print(f"GIF créé : {chemin_gif}")
