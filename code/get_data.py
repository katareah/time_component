import requests
import os

def get_files_from_repo(path, repo="Mcompetitions/M4-methods", branch="master"):
    """ Fetch file details from a specified repo and branch """
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    response = requests.get(api_url)
    return response.json()  # Returns a list of dictionaries, one for each file

def download_file(file_info):
    """ Download a single file using the 'download_url' provided in the file_info """
    response = requests.get(file_info['download_url'])
    if response.status_code == 200:
        # Create the local path, if directory does not exist, create it
        os.makedirs(os.path.dirname(file_info['path']), exist_ok=True)
        with open(file_info['path'], 'wb') as file:
            file.write(response.content)
            print(f"Downloaded {file_info['path']}")

def main():
    # Path to the dataset directory on GitHub
    path = "Dataset/Test"
    
    # Fetch file list from GitHub
    files = get_files_from_repo(path)

    # Download each file
    for file_info in files:
        if file_info['type'] == 'file':  # Check if it is a file and not a directory
            download_file(file_info)
        else:
            print(f"Skipped {file_info['path']}, it's not a file.")

if __name__ == "__main__":
    main()
