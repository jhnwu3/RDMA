import pandas as pd
# data_manager.py
from hporag.config import Config
class DataManager:
    @staticmethod
    def load_clinical_notes():
        """Load clinical notes from user input or file."""
        user_input = input("Do you want to provide clinical notes directly (yes/no)? ").strip().lower()
        
        if user_input == 'yes':
            return DataManager._load_from_input()
        else:
            return DataManager._load_from_file()

    @staticmethod
    def _load_from_input():
        """Load clinical notes from direct user input."""
        clinical_notes = []
        while True:
            note = input("Enter clinical note (or type 'done' to finish): ").strip()
            if note.lower() == 'done':
                break
            clinical_notes.append(note)
            
        df = pd.DataFrame({'clinical_note': clinical_notes})
        df['patient_id'] = range(1, len(df) + 1)
        return df

    @staticmethod
    def _load_from_file():
        """Load clinical notes from a CSV file."""
        while True:
            input_file = input("Enter the filename of the CSV containing clinical notes: ").strip()
            if not input_file.lower().endswith('.csv'):
                print("The file must have a .csv extension.")
                continue

            try:
                df = pd.read_csv(input_file)
                
                if 'patient_id' in df.columns:
                    df['patient_id'] = df['patient_id']
                elif 'case number' in df.columns:
                    df['patient_id'] = df['case number']
                else:
                    df['patient_id'] = range(1, len(df) + 1)
                
                return df
                
            except FileNotFoundError:
                print("File not found. Please ensure the file exists and the path is correct.")
            except pd.errors.EmptyDataError:
                print("The file is empty. Please provide a valid CSV file with data.")
            except Exception as e:
                print(f"An error occurred: {e}")

    @staticmethod
    def save_results(results_df, output_path=None):
        """Save the processed results."""
        if output_path is None:
            output_path = input("Enter the output file path (with .csv extension): ").strip()
            
        results_df.to_csv(output_path, index=False)
        Config.timestamped_print(f"Results saved to {output_path}")