import os
import pandas as pd
from torch.utils.data import Dataset

# FUNZIONE PER GENERARE LISTE DEL DATASET
def generate_dataset_components(parquet_file, text_output_folder):
    """
    Dato un file parquet con domande e nomi immagine,
    genera le liste di:
    - questions
    - percorsi ai parquet dei contesti perfetti (UNet)
    - percorsi ai file .txt dove salvare i prompt combinati
    """
    perfect_context_folder = '.../BiasesProject/DataSet/UNET_context/'

    questions = parquet_file['question'].tolist()
    perfect_contexts = [
        os.path.join(perfect_context_folder, f"{img_name}.parquet")
        for img_name in parquet_file['img_name']
    ]
    txt_filenames = [
        os.path.join(text_output_folder, f"{img_name}.txt")
        for img_name in parquet_file['index']
    ]

    print("Numero file di testo da generare:", len(txt_filenames))
    return questions, perfect_contexts, txt_filenames

# CLASSE DATASET VQA
class VQADataset(Dataset):
    def __init__(self, questions, perfect_contexts, txt_filenames):
        self.questions = questions
        self.perfect_contexts = perfect_contexts
        self.txt_filenames = txt_filenames

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        """
        Restituisce il testo combinato Question + Table
        e salva il prompt in un file .txt
        """
        question = self.questions[idx]
        df_context = pd.read_parquet(self.perfect_contexts[idx])

        # Genera i token per ciascuna cella della tabella
        patch_tokens = []
        for row_name in df_context.index:
            for col_name in df_context.columns:
                value = df_context.loc[row_name, col_name]
                if value != 0:
                    token = f"({col_name},{row_name}):{value}"
                    patch_tokens.append(token)

        patch_tokens_str = ",".join(patch_tokens)
        combined_text = f"Question: {question}; Table: {patch_tokens_str}"

        # Salva il testo in un file .txt
        txt_filename = self.txt_filenames[idx]
        os.makedirs(os.path.dirname(txt_filename), exist_ok=True)
        with open(txt_filename, 'w') as f:
            f.write(combined_text)

        return combined_text

# SCRIPT PRINCIPALE
if __name__ == "__main__":
    parquet_files = [
        #'/path/to/train.parquet',
        #'/path/to/validation.parquet',
        '.../BiasesProject/DataSet/parquet/qafinal/dataset_balanced_test.parquet'
    ]

    text_output_folder = ".../BiasesProject/DataSet/UNET_text_context/" # either UNET, Segformer, DOFA or perfect context text folder results 
    os.makedirs(text_output_folder, exist_ok=True)

    for parquet_file_path in parquet_files:
        print(f"Processing: {parquet_file_path}")

        parquet_file = pd.read_parquet(parquet_file_path)
        questions, perfect_contexts, txt_filenames = generate_dataset_components(
            parquet_file, text_output_folder
        )

        vqa_dataset = VQADataset(questions, perfect_contexts, txt_filenames)

        # Genera e salva tutti i file .txt
        for idx in range(len(vqa_dataset)):
            _ = vqa_dataset[idx]

        print(f"Finished processing: {parquet_file_path}")
