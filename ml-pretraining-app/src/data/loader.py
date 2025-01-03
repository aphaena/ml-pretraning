import os
from typing import List, Dict
import pdfplumber
from transformers import AutoTokenizer
import torch
import requests
import json
import pickle
import datetime

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
        self.session = requests.Session()

        self.headers = {
            "Content-Type": "application/json", "Connection": "keep-alive" # Ensure the connection is kept alive
        }

        try:
            # Utiliser le tokenizer spécifié dans la configuration
            tokenizer_name = config['lmstudio']['tokenizer_name']
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True
            )
            print(f"Tokenizer '{tokenizer_name}' chargé avec succès")

            # Vérifier si LM Studio est disponible
            response = self.session.get(f"{config['lmstudio']['api_url']}/models", headers=self.headers)
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

        # Ajout du répertoire de sauvegarde
        self.save_dir = os.path.join(os.getcwd(), 'data', 'training_data')
        os.makedirs(self.save_dir, exist_ok=True)

    def process_with_lmstudio(self, text: str) -> str:
        """Utilise LM Studio pour le traitement du texte"""
        try:
            response = self.session.post(
                f"{self.api_url}/chat/completions",
                headers=self.headers,
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
        
    def process_text_with_instructions(self, text: str) -> List[dict]:
        """Process text with all configured instructions"""
        results = []
        for instruction in self.config['prompts']:
            result = self.process_with_lmstudio(text, instruction['instruction'])
            results.append({
                'instruction': instruction['instruction'],
                'input': text,
                'output': result
            })
        return results        

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
                print(f"Traitement du fichier: {filename}")
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
                        # Pour chaque instruction dans la configuration
                        for prompt in self.config['prompts']:
                            instruction = prompt['instruction']
                            # Création de la requête pour LM Studio
                            prompt_text = f"{instruction}\n\nTexte: {chunk}"
                            # Obtention de la réponse
                            response = self.process_with_lmstudio(prompt_text)
                            # Ajout à l'ensemble d'entraînement
                            training_data.append({
                                "instruction": instruction,
                                "input": chunk,
                                "output": response
                            })
                            print(f"Instruction traitée: {instruction[:50]}...")
        
        print(f"Nombre total d'exemples générés: {len(training_data)}")

        # Sauvegarde automatique des données générées
        self.save_training_data(training_data)
        
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

    def save_training_data(self, training_data: List[dict], filename: str = None) -> str:
        """Sauvegarde les données d'entraînement"""
        if filename is None:
            # Création d'un nom de fichier avec la date et l'heure
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}"

        # Création des chemins de fichiers
        json_path = os.path.join(self.save_dir, f"{filename}.json")
        pickle_path = os.path.join(self.save_dir, f"{filename}.pkl")

        # Sauvegarde au format JSON (lisible)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        # Sauvegarde au format Pickle (complet avec métadonnées)
        metadata = {
            'timestamp': timestamp,
            'config': self.config,
            'model_name': self.config['model']['name'],
            'num_examples': len(training_data)
        }
        with open(pickle_path, 'wb') as f:
            pickle.dump({'data': training_data, 'metadata': metadata}, f)

        print(f"Données sauvegardées dans:\n- {json_path}\n- {pickle_path}")
        return filename

    def load_training_data(self, filename: str) -> List[dict]:
        """Charge les données d'entraînement sauvegardées"""
        pickle_path = os.path.join(self.save_dir, f"{filename}.pkl")
        
        try:
            with open(pickle_path, 'rb') as f:
                saved_data = pickle.load(f)
                print(f"Métadonnées du fichier chargé:")
                for key, value in saved_data['metadata'].items():
                    print(f"- {key}: {value}")
                return saved_data['data']
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return None

    def list_available_training_data(self) -> List[Dict]:
        """Liste tous les fichiers d'entraînement disponibles"""
        available_files = []
        for filename in os.listdir(self.save_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(self.save_dir, filename)
                try:
                    with open(file_path, 'rb') as f:
                        saved_data = pickle.load(f)
                        available_files.append({
                            'filename': filename,
                            'metadata': saved_data['metadata'],
                            'path': file_path
                        })
                except Exception as e:
                    print(f"Erreur lors de la lecture de {filename}: {e}")
        return available_files

    def load_all_training_data(self) -> List[dict]:
        """Charge et combine toutes les données d'entraînement disponibles"""
        all_data = []
        available_files = self.list_available_training_data()
        
        if not available_files:
            print("Aucun fichier d'entraînement trouvé.")
            return None
            
        print("\nFichiers d'entraînement disponibles:")
        for idx, file_info in enumerate(available_files):
            print(f"\n{idx + 1}. {file_info['filename']}")
            print("   Métadonnées:")
            for key, value in file_info['metadata'].items():
                if key != 'config':  # Skip printing full config
                    print(f"   - {key}: {value}")

        response = input("\nVoulez-vous charger ces données? (o/n): ").lower()
        if response != 'o':
            print("Chargement des données annulé.")
            return None

        print("\nChargement des données...")
        for file_info in available_files:
            try:
                with open(file_info['path'], 'rb') as f:
                    saved_data = pickle.load(f)
                    all_data.extend(saved_data['data'])
                print(f"✓ {file_info['filename']} chargé ({len(saved_data['data'])} exemples)")
            except Exception as e:
                print(f"✗ Erreur lors du chargement de {file_info['filename']}: {e}")

        print(f"\nTotal: {len(all_data)} exemples chargés depuis {len(available_files)} fichiers")
        return all_data
    

    def upload_to_lmstudio(self, data, api_url, headers):
        """Utilise LM Studio pour traiter des données via /v1/chat/completions"""
        for idx, example in enumerate(data):
            try:
                response = self.session.post(
                    f"{api_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": "default",  # Remplacez par le modèle souhaité, si applicable
                        "messages": [                            
                            {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"}
                        ],
                        "max_tokens": 500,
                        "temperature": 0.7
                    },
                    timeout=10  # Timeout pour éviter les blocages
                )
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result:
                        example["output"] = result["choices"][0]["message"]["content"]
                        print(f"Exemple {idx + 1}/{len(data)} traité avec succès.")
                    else:
                        print(f"Réponse inattendue pour l'exemple {idx + 1}/{len(data)} : {result}")
                else:
                    print(f"Erreur pour l'exemple {idx + 1}/{len(data)} : {response.status_code}, {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"Erreur réseau pour l'exemple {idx + 1}/{len(data)} : {e}")
            except KeyError as e:
                print(f"Clé manquante dans la réponse pour l'exemple {idx + 1}/{len(data)} : {e}")
