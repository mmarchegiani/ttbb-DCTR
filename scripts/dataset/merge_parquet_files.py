import argparse
import awkward as ak
import pyarrow.parquet as pq

def merge_parquet_files(file1: str, file2: str, output_file: str):
    """
    Merges two Parquet files into one using Awkward Array and saves the result as a new Parquet file.

    Parameters:
        file1 (str): Path to the first Parquet file.
        file2 (str): Path to the second Parquet file.
        output_file (str): Path to save the merged Parquet file.
    """
    # Read the Parquet files as Awkward Arrays
    array1 = ak.from_parquet(file1)
    array2 = ak.from_parquet(file2)

    # Merge the arrays
    merged_array = ak.concatenate([array1, array2])

    # Save the merged array to a new Parquet file
    output_file = output_file.replace(".parquet", f"_{len(merged_array)}.parquet")
    ak.to_parquet(merged_array, output_file)

    print(f"Merged Parquet file saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two Parquet files into one.")
    parser.add_argument("file1", type=str, help="Path to the first Parquet file.")
    parser.add_argument("file2", type=str, help="Path to the second Parquet file.")
    parser.add_argument("-o", "--output", type=str, help="Path to save the merged Parquet file.")
    args = parser.parse_args()

    merge_parquet_files(args.file1, args.file2, args.output)
