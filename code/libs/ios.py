import os
import glob
import pandas as pd
from datetime import datetime
import json

FN_AGENTS = 'agents.txt'
FN_PROXIES = 'http_proxies.txt'

######################################################
#### PATHS
######################################################

def path_exists(path):
    """
    Checks if a given path exists.
    
    :param path: Path to check.
    :return: True if the path exists, False otherwise.
    """
    return os.path.exists(path)

def path_join(*paths):
   return os.path.join(*paths)

def get_path(fn):
   return os.path.dirname(fn)

def validate_path(path, **kwargs):
    """
    Validates a directory path, creating it if it doesn't exist.

    :param path: The directory path to validate.
    :param kwargs: Additional arguments to pass to os.makedirs.
    """
    try:
      if not os.path.exists(path):
          os.makedirs(path, **kwargs)
          printf(f"[INFO] Valid path: {path}")
    except Exception as ex:
       printf(f"[ERROR] {ex}")
       

def listdir(path, pattern=None, ext=None):
    """
    Lists files in a directory with optional pattern and extension filtering.
    
    :param path: Directory path to list files from.
    :param pattern: Optional regex pattern to match filenames.
    :param ext: Optional file extension to filter by.
    :return: List of matching file paths.
    """
    if not os.path.exists(path):
        return []
    
    files = os.listdir(path)

    if pattern:
        files = glob.glob(os.path.join(path, f"*{pattern}*"))
  
    if ext:
        files = [f for f in files if f.endswith(ext)]
    
    print(f'[INFO] Found {len(files)} files in {path}.')
    return files


######################################################
#### IO
######################################################


def save_json(obj, fn):
    try:
      with open(fn, 'w') as f:
          json.dump(obj, f)
    except Exception as e:
      print(f'Error: {e}')

def read_json(fn):
    try:
      with open(fn, 'r') as f:
          return json.load(f)
    except Exception as e:
      print(f'Error: {e}')
    return None
    
def to_csv(df, fn, **kwargs):
    try:
      df.to_csv(fn, **kwargs)
    except Exception as e:
      print(f'Error: {e}')

def load_csv(fn, **kwargs):
  if fn is not None and os.path.exists(fn):
    return pd.read_csv(fn, **kwargs)
  return None

def load_excel(fn):
    if fn is not None and os.path.exists(fn):
       return pd.read_excel(fn)
    return None

def read_txt_to_lst(filename):
    """
    Reads a file and returns a list of lines.

    :param filename: The path to the file to be read.
    :return: A list where each item is a line from the file.
    """
    try:
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        return lines
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


######################################################
#### IO-Dataframes
######################################################

def load_json_files_to_dataframe(directory, include_fn=False):
    """
    Reads all JSON files in a directory and converts them into a Pandas DataFrame.

    :param directory: Path to the folder containing JSON files.
    :return: Pandas DataFrame with concatenated data from all JSON files.
    """
    data_list = []

    files = glob.glob(directory)

    # Loop through all files in the directory
    for file_path in files:
        filename = os.path.basename(file_path)
        if file_path.endswith(".json"):  # Process only JSON files
            # file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)  # Load JSON content

                    if include_fn:
                       data.update({'fn': file_path})

                    if isinstance(data, dict):
                        data_list.append(data)  # Append dictionary data
                    elif isinstance(data, list):
                        data_list.extend(data)  # Extend list of dictionaries
                except json.JSONDecodeError:
                    print(f"Error reading {filename}, skipping...")

    # Convert list of dictionaries to DataFrame
    return pd.DataFrame(data_list)

def merge_dataframes_from_folder(path):
  files = glob.glob(os.path.join(path, 'chunk_*.csv'))
  print(f'[INFO] reading {len(files)} files.')
  df = pd.concat([pd.read_csv(fn, index_col=0) for fn in files])
  return df


######################################################
#### STDOUT
######################################################

def printf(message: str):
    """
    Prints a message with a timestamp in front.

    Args:
        message (str): The message to print.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")



######################################################
#### Others
######################################################

def read_agents(path):
  if path is not None:
    return read_txt_to_lst(os.path.join(path, FN_AGENTS))
  return None

def read_proxies(path):
  if path is not None:
    return read_txt_to_lst(os.path.join(path, FN_PROXIES))
  return None


