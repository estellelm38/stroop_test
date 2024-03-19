import cv2
import numpy as np
import cv2
import cv2
import easyocr
import torch
import torch.nn as nn
import torch.optim as optim

#ici on mettra tous les paramètres utilisés dans les différentes fonctions 

class Agent:
    
    #fonction qui permet de récupérer la couleur dominante de l'image à partir de la teinte
    def get_color(hue_value):

        color_thresholds = {
            "Rouge": ([0, 32], [160, 170]),
            "Jaune": ([23, 38],),
            "Vert": ([40, 124],),
            "Bleue": ([125, 200],),
        }

        for color_name, hue_ranges in color_thresholds.items():
            for hue_range in hue_ranges:
                if hue_value in range(hue_range[0], hue_range[-1] + 1):
                    dominant_color_name = color_name
                    break

        return dominant_color_name

    def __init__(self, grayscale_image_path, color_image_path):
        self.grayscale_image_path = grayscale_image_path
        self.color_image_path = color_image_path

    def identify_color(self):
        color_image = cv2.imread(self.color_image_path)

        # erreur de chargement de l'image
        if color_image is None:
            print("Erreur lors du chargement de l'image.")
            return None

        # converti la couleur de l'image en HSV
        color_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Calcul l'histogramme des valeurs HSV
        hist = cv2.calcHist([color_hsv], [0], None, [256], [0, 256])

        # Find the dominant color bin
        dominant_bin = np.argmax(hist)
        dominant_hue = int(dominant_bin * 2)  # Convert bin to hue value

        # défini le nom de la couleur en fonction de leur valeur HSV
        color_thresholds = {
            "Rouge": ([0, 32], [160, 170]),
            "Jaune": ([23, 38],),
            "Vert": ([40, 124],),
            "Bleue": ([125, 200],),
        }
        print(dominant_hue)
        # Match the dominant hue to a color range
        dominant_color_name = "Undefined"
        for color_name, hue_ranges in color_thresholds.items():
            for hue_range in hue_ranges:
                if dominant_hue in range(hue_range[0], hue_range[-1] + 1):
                    dominant_color_name = color_name
                    break

        return dominant_color_name
    
    #fonction permettant
    def identify_word():

        # Chemin vers l'image
        image_path = '/home/estelle/robotlearn/projetbl/pictures/testmot.jpg'

        # Charger l'image avec OpenCV
        image = cv2.imread(image_path)

        # Convertir l'image en RGB (easyocr exige les images en RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Utiliser EasyOCR pour reconnaître le texte
        reader = easyocr.Reader(['en'])  # Spécifiez les langues que vous voulez reconnaître ici
        result = reader.readtext(image_rgb)

        # Récupérer le texte extrait
        extracted_text = [entry[1] for entry in result]

        # Afficher le texte extrait
        return extracted_text


    print(identify_word())

# Définition du réseau de neurones
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Une seule sortie pour la récompense

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Fonction pour comparer deux entrées et attribuer une récompense
    def comparer_entrees(reseau, mot, couleur, critere, optimizer):
        # Encodage des mots et couleurs en tensors (représentation numérique)
        mot_tensor = torch.Tensor([ord(c) for c in mot]).unsqueeze(0)  # Encodage du mot en utilisant les valeurs ASCII
        couleur_tensor = torch.Tensor([ord(c) for c in couleur]).unsqueeze(0)  # Encodage de la couleur en ASCII
        
        # Passage des données dans le réseau de neurones
        output_mot = reseau(mot_tensor)
        output_couleur = reseau(couleur_tensor)
        
        # Calcul de la perte (loss)
        loss = critere(output_mot, output_couleur)
        
        # Mise à jour des poids du réseau de neurones
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calcul de la récompense
        reward = 1.0 if torch.argmax(output_mot) == torch.argmax(output_couleur) else 0.0
        
        return reward

# Définition des paramètres
input_size = 7  # Taille de l'entrée (nombre de caractères dans le mot ou la couleur)
hidden_size = 128  # Taille de la couche cachée

# Initialisation du réseau de neurones
reseau = SimpleNet(input_size, hidden_size)
critere = nn.MSELoss()  # Utilisation de la Mean Squared Error comme fonction de perte
optimizer = optim.SGD(reseau.parameters(), lr=0.01)  # Optimiseur Stochastic Gradient Descent (SGD) avec un learning rate de 0.01

# Exemple d'utilisation

mot = identify_word()
couleur = identify_color()

reward = comparer_entrees(reseau, mot, couleur, critere, optimizer)
print(f"Récompense: {reward}")



# grayscale_image_path = '/home/estelle/robotlearn/projetbl/pictures/herbe.jpg'
# color_image_path = '/home/estelle/robotlearn/projetbl/pictures/herbe.jpg'

# color_identifier = Agent(grayscale_image_path, color_image_path)
# dominant_color = color_identifier.identify_color()
# print("La couleur dominante dans l'image en couleur est :", dominant_color)

# grayscale_image_path = '/home/estelle/robotlearn/projetbl/pictures/citrouilles.jpg'
# color_image_path = '/home/estelle/robotlearn/projetbl/pictures/citrouilles.jpg'

# color_identifier = Agent(grayscale_image_path, color_image_path)
# dominant_color = color_identifier.identify_color()
# print("La couleur dominante dans l'image en couleur est :", dominant_color)

# grayscale_image_path = '/home/estelle/robotlearn/projetbl/pictures/ocean.jpg'
# color_image_path = '/home/estelle/robotlearn/projetbl/pictures/ocean.jpg'

# color_identifier = Agent(grayscale_image_path, color_image_path)
# dominant_color = color_identifier.identify_color()
# print("La couleur dominante dans l'image en couleur est :", dominant_color)

# grayscale_image_path = '/home/estelle/robotlearn/projetbl/pictures/testmot.jpg'
# color_image_path = '/home/estelle/robotlearn/projetbl/pictures/testmot.jpg'

# color_identifier = Agent(grayscale_image_path, color_image_path)
# dominant_color = color_identifier.identify_color()
# print("La couleur dominante dans l'image en couleur est :", dominant_color)

# grayscale_image_path = '/home/estelle/robotlearn/projetbl/pictures/noir.jpg'
# color_image_path = '/home/estelle/robotlearn/projetbl/pictures/noir.jpg'

# color_identifier = Agent(grayscale_image_path, color_image_path)
# dominant_color = color_identifier.identify_color()
# print("La couleur dominante dans l'image en couleur est :", dominant_color)