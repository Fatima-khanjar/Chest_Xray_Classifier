import zipfile
import os
##step1:unzip the file of dataset

# Path to the downloaded zip file
zip_file_path = r'D:\fall semster 23-24\Machine Learning\ML Project\covid19-radiography-database.zip'

# Directory where you want to extract the contents
extracted_dir = r'D:\fall semster 23-24\Machine Learning\ML Project'

# Create the target directory if it doesn't exist
os.makedirs(extracted_dir, exist_ok=True)

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents to the target directory
    zip_ref.extractall(extracted_dir)

print(f'Dataset extracted to: {extracted_dir}')