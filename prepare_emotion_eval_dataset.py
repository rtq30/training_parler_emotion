import json
import pandas as pd
from datasets import Dataset, DatasetDict
import argparse


def create_emotion_evaluation_dataset(json_file_path, output_dataset_path):
    """
    Convert JSON evaluation data to HuggingFace Dataset format.
    """
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    # Replace None/null values with empty strings for string columns
    # Hardy: I modified the string_columns to match the training data set
    # string_columns = ['description', 'text', 'emotion', 'gender', 'background_noise',
    #                   'pitch', 'rate', 'test_category']
    string_columns = ['text_description', 'text', 'style', 'gender', 'noise',
                      'pitch', 'speaking_rate', 'test_category']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Create HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    # Create a DatasetDict with just eval split
    dataset_dict = DatasetDict({
        'eval': dataset
    })

    # Save to disk
    dataset_dict.save_to_disk(output_dataset_path)

    # Print summary statistics
    print(f"Created evaluation dataset with {len(dataset)} samples")
    print("\nEmotion distribution:")
    # Hardy: I cried.
    # emotion_counts = df['emotion'].value_counts()
    emotion_counts = df['style'].value_counts()

    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count}")

    print(f"\nDataset saved to: {output_dataset_path}")

    return dataset_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to HuggingFace Dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output dataset directory")

    args = parser.parse_args()
    create_emotion_evaluation_dataset(args.input, args.output)