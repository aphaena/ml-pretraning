from src.data.loader import DataLoader  # Modifier le chemin d'import
from src.models.mistral_trainer import MistralTrainer
import yaml
import os

def main():
    # Charger la configuration
    with open('src/config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Créer le répertoire pour les PDFs s'il n'existe pas
    pdf_dir = os.path.join(os.getcwd(), 'data', 'pdfs')  # Chemin absolu
    os.makedirs(pdf_dir, exist_ok=True)

    print("\n=== Configuration ===")
    print(f"PDF Directory: {pdf_dir}")
    print(f"LM Studio API: {config['lmstudio']['api_url']}")
    print("===================\n")

    print("1. Vérification de LM Studio...")
    input("Assurez-vous que LM Studio est lancé. Appuyez sur Entrée pour continuer...")

    # Initialiser le DataLoader avec la configuration complète
    loader = DataLoader(
        model_name=config['model']['name'],
        pdf_dir=pdf_dir,
        config=config
    )
    
    # Proposer de charger les données existantes
    print("\n=== Données d'entraînement ===")
    training_data = loader.load_all_training_data()
    
    if training_data is None:
        print("\nGénération de nouvelles données...")
        training_data = loader.load_data()  # Génère et sauvegarde de nouvelles données

    if training_data:
        print("\nExemple d'entraînement :")
        print("Instruction:", training_data[0]['instruction'])
        print("Input:", training_data[0]['input'][:200], "...")
        print("Output:", training_data[0]['output'])

        # Traitement des données avec LM Studio
        print("\nTraitement des données dans LM Studio...")
        loader.upload_to_lmstudio(
            data=training_data,
            api_url=config['lmstudio']['api_url'],
            headers={"Content-Type": "application/json", "Connection": "keep-alive"}
        )   
    else:
        print("Aucune donnée trouvée ou générée.")

if __name__ == "__main__":
    main()
