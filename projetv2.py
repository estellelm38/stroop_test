import cv2
import numpy as np
import cv2
import easyocr
import torch
import torch.nn as nn
import torch.optim as optim
import os

class Agent:

    def __init__(self, color_image_path):
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
            "ROUGE": ([0,30],), #(0,100,100)
            "JAUNE": ([31,90],), #(60,100,100)
            "VERT": ([91,150],), #(103,76,47) -> (120,100,100)
            "BLEU": ([151,240],), #(240,100,100)
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

    def identify_word(self, image_path):
        # Charger l'image avec OpenCV
        image = cv2.imread(image_path)

        # Gestion d'une erreur de chargement de l'image
        if image is None:
            print(f"Erreur lors du chargement de l'image : {image_path}")
            return None

        # Convertir l'image en virgule flottante et normaliser les valeurs de pixel
        image = image.astype(np.float32) / 255.0

        # Convertir l'image en RGB (easyocr exige les images en RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Utiliser EasyOCR pour reconnaître le texte
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_rgb)

        # Vérifier si le texte a été extrait correctement
        if result:
            # Récupérer le texte extrait
            extracted_text = [entry[1] for entry in result]

            # Vérifier si la liste extracted_text n'est pas vide
            if extracted_text:
                # Afficher le texte extrait
                mot = extracted_text[0]
                return mot
            else:
                print(f"Aucun texte n'a été extrait de l'image : {image_path}")
        else:
            print(f"Aucun résultat d'extraction de texte pour l'image : {image_path}")

        return None
    
    def comparer_entrees(self, critere, optimizer, reseau):
        # Charger l'image avec OpenCV
        image = cv2.imread(self.color_image_path)

        # Gestion d'une erreur de chargement de l'image
        if image is None:
            print(f"Erreur lors du chargement de l'image : {self.color_image_path}")
            return None

        # Normaliser les valeurs de pixel de l'image
        image = image / 255.0

        # Convertir l'image en RGB (easyocr exige les images en RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Utiliser EasyOCR pour reconnaître le texte
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_rgb)

        # Récupérer le texte extrait
        extracted_text = [entry[1] for entry in result]

        # Vérifier si le texte a été extrait correctement et n'est pas vide
        if extracted_text:
            # Afficher le texte extrait
            mot = extracted_text[0]

            # Encodage des mots en tensors (représentation numérique)
            # Utilisez une représentation appropriée pour votre modèle (exemple : one-hot encoding)
            # Dans l'exemple ci-dessous, on utilise simplement la valeur ASCII des caractères
            mot_tensor = torch.Tensor([ord(c) for c in mot]).unsqueeze(0)

            # Passage des données dans le réseau de neurones
            output_mot = reseau(mot_tensor)

            # Calcul de la perte (loss)
            loss = critere(output_mot, reseau(mot_tensor))

            # Mise à jour des poids du réseau de neurones
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calcul de la récompense
            reward = 1.0 if torch.argmax(output_mot) == torch.argmax(reseau(mot_tensor)) else 0.0

            print(f"Récompense pour le mot {mot}: {reward}")
            return loss.item(), reward
        else:
            print(f"Aucun texte n'a été extrait de l'image : {self.color_image_path}")
            return None

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

class Training:
    def __init__(self, folder_path, reseau):
        self.folder_path = folder_path
        self.reseau = reseau

    def calculate_total_reward(self, critere, optimizer):
        total_reward = 0

        # Parcourez chaque fichier dans le dossier
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".jpg"):  # Assurez-vous de filtrer les fichiers image
                image_path = os.path.join(self.folder_path, filename)

                # Utilisez un nouvel agent pour chaque image
                agent = Agent(image_path)

                # Identifiez le mot et la couleur pour chaque image
                mot = agent.identify_word(image_path)
                couleur = agent.identify_color()

                # Calculez la récompense pour cette image spécifique
                reward = agent.comparer_entrees(critere, optimizer, self.reseau)
                total_reward += reward

        return total_reward


# Initialisation du réseau de neurones, déclaration des paramètres et exemple d'utilisation
input_size = 4
hidden_size = 128

reseau = SimpleNet(input_size, hidden_size)
critere = nn.MSELoss()
optimizer = optim.SGD(reseau.parameters(), lr=0.01)

# Dossier contenant les images pour l'entraînement
folder_path = '/home/estelle/robotlearn/projetbl/Donnees/Congruent'

# Utilisation de la classe Training
training_instance = Training(folder_path, reseau)
total_reward = training_instance.calculate_total_reward(critere, optimizer)
print(f"Récompense totale pour le dossier : {total_reward}")
