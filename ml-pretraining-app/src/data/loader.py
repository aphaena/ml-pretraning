import os
from typing import List, Dict
import pdfplumber
from transformers import AutoTokenizer
import torch
import requests
import json

class DataLoader:
    def __init__(self, model_name=None, pdf_dir="data/pdfs", config=None):
        """
        Initialize the DataLoader
        Args:
            model_name: Nom du modèle de référence pour le tokenizer
            pdf_dir: Répertoire contenant les PDFs
            config: Configuration complète
        """
        if config is None:
            raise ValueError("Le paramètre config est requis")

        self.config = config
        self.pdf_dir = pdf_dir
        self.max_length = 2048

        try:
            # Utiliser le tokenizer spécifié dans la configuration
            tokenizer_name = config['lmstudio']['tokenizer_name']
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True
            )
            print(f"Tokenizer '{tokenizer_name}' chargé avec succès")

            # Vérifier si LM Studio est disponible
            response = requests.get(f"{config['lmstudio']['api_url']}/models")
            if response.status_code == 200:
                print("LM Studio détecté en local")
                self.use_local = True
                self.api_url = config['lmstudio']['api_url']
            else:
                raise Exception("LM Studio n'est pas accessible")

        except Exception as e:
            print(f"Erreur: {e}")
            print("Assurez-vous que LM Studio est lancé et accessible sur http://localhost:1234")
            raise

    def process_with_lmstudio(self, text: str) -> str:
        """Utilise LM Studio pour le traitement du texte"""
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": text}],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=self.config['lmstudio']['timeout']
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"Erreur LM Studio: {response.status_code}")
                return text
        except Exception as e:
            print(f"Erreur lors du traitement avec LM Studio: {e}")
            return text

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrait le texte d'un fichier PDF."""
        # Variable pour stocker le texte extrait
        text = ""
        try:
            # Ouverture du fichier PDF avec pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                # Parcours de chaque page du PDF
                for page in pdf.pages:
                    # Extraction du texte et ajout d'un saut de ligne
                    text += page.extract_text() + "\n"
        except Exception as e:
            # Gestion des erreurs lors de la lecture du PDF
            print(f"Erreur lors de la lecture du PDF {pdf_path}: {e}")
        return text

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Découpe le texte en chunks de taille similaire."""
        # Séparation du texte en mots
        words = text.split()
        # Liste pour stocker les chunks
        chunks = []
        # Liste temporaire pour le chunk en cours
        current_chunk = []
        # Compteur de taille du chunk actuel
        current_size = 0
        
        # Parcours de chaque mot du texte
        for word in words:
            # Ajout du mot au chunk actuel
            current_chunk.append(word)
            # Mise à jour de la taille du chunk (+1 pour l'espace)
            current_size += len(word) + 1
            
            # Si le chunk atteint la taille maximale
            if current_size >= chunk_size:
                # Ajout du chunk à la liste des chunks
                chunks.append(" ".join(current_chunk))
                # Réinitialisation du chunk en cours
                current_chunk = []
                current_size = 0
        
        # Traitement du dernier chunk s'il existe
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def summarize_chunk(self, chunk: str, max_length: int = 200) -> str:
        """Crée un résumé en utilisant LM Studio"""
        prompt = f"Résume ce texte en quelques phrases: {chunk}"
        return self.process_with_lmstudio(prompt)

    def extract_key_points(self, chunk: str) -> str:
        """Extrait les points clés en utilisant LM Studio"""
        prompt = f"Extrais les points clés de ce texte: {chunk}"
        return self.process_with_lmstudio(prompt)

    def load_data(self) -> List[dict]:
        """Charge et prépare les données depuis les fichiers PDF."""
        # Liste pour stocker les données d'entraînement
        training_data = []
        
        # Parcours des fichiers dans le répertoire PDF
        for filename in os.listdir(self.pdf_dir):
            # Vérification de l'extension PDF
            if filename.endswith('.pdf'):
                # Construction du chemin complet du fichier
                pdf_path = os.path.join(self.pdf_dir, filename)
                # Extraction du texte du PDF
                text = self.extract_text_from_pdf(pdf_path)
                # Découpage du texte en chunks
                chunks = self.chunk_text(text)
                
                # Création des exemples d'entraînement pour chaque chunk
                for chunk in chunks:
                    # Vérification que le chunk n'est pas vide
                    if chunk.strip():
                        # Ajout des exemples d'entraînement avec différentes instructions
                        training_data.extend([
                            {
                                "instruction": "Résume ce passage du document.",
                                "input": chunk,
                                "output": self.summarize_chunk(chunk)
                            },
                            {
                                "instruction": "Extrais les informations principales de ce passage.",
                                "input": chunk,
                                "output": self.extract_key_points(chunk)
                            }
                        ])
        
        return training_data

    def prepare_data(self, examples: List[dict]) -> dict:
        """Prépare les données au format Mistral pour le fine-tuning."""
        # Formatage des textes selon le format attendu par Mistral
        formatted_texts = [
            f"[INST] {ex['instruction']}\n\n{ex['input']} [/INST] {ex['output']}"
            for ex in examples
        ]
        
        # Tokenization des textes avec padding et troncature
        return self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def get_data(self):
        """Charge et prépare les données en une seule étape."""
        # Chargement des exemples
        examples = self.load_data()
        # Préparation des données pour l'entraînement
        prepared_data = self.prepare_data(examples)
        return prepared_data