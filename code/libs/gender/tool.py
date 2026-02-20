import pandas as pd

import ios

GENDER_COLUMN = 'gender'
GIVENNAME_COLUMN = 'givenname'

class Tool(object):
    """
    A class for various text processing tasks.
    """

    def __init__(self):
        self.givennames_list: list[str] | pd.DataFrame | None = None
        self.givennames_df : pd.DataFrame | None = None

    def load_givennames_list(self, filepath: str|None, lst: list[str]|None):
        """
        Load a list of Chinese given names from a text file or a provided list into self.givennames_list.
        
        Args:
            filepath (str): Path to the text file containing given names.
            lst (list): A list of given names.
        """
        if filepath:
            self.givennames_list = ios.read_txt_to_lst(filepath)
        elif lst:
            self.givennames_list = lst
        else:
            raise ValueError("No filepath or list provided for given names.")
        
        self.givennames_df = pd.DataFrame(self.givennames_list, columns=[GIVENNAME_COLUMN])

    def load_givennames_df(self, filepath: str|None, df: pd.DataFrame|None):
        """
        Load a DataFrame of Chinese given names from a CSV file or a provided DataFrame into self.givennames_df.
        
        Args:
            filepath (str): Path to the CSV file containing given names.
            df (DataFrame): A DataFrame of given names.
        """
        if filepath:
            self.givennames_df = ios.load_csv(filepath)
        elif df is not None:
            self.givennames_df = df
        else:
            raise ValueError("No filepath or DataFrame provided for given names.")

        if self.givennames_df is not None and GIVENNAME_COLUMN not in self.givennames_df.columns:
            raise ValueError(f"The DataFrame must contain a '{GIVENNAME_COLUMN}' column.")
        else:
            self.givennames_list = self.givennames_df[GIVENNAME_COLUMN].tolist()

    def _infer_gender_by_givenname(self, name: str) -> str:
        """
        Infer the gender from a given Chinese name.
        """
        pass
    
    def infer_gender_by_givenname(self, name: str, err: bool) -> str:
        """
        Infer the gender from a given Chinese name.
        """
        if name is None or len(name) == 0 or name == '':
            if err:
                raise ValueError("Name is empty or None")
            else:
                return None
            
        return self._infer_gender_by_givenname(name)
        
    def infer_gender_by_givennames(self):
        """
        Infer the gender from a given Chinese name using the given names list.
        """
        if self.givennames_df is None:
            raise ValueError("Given names DataFrame is not loaded.")

        self.givennames_df.loc[:, GENDER_COLUMN] = None

    def save_gender_inference(self, filepath: str):
        """
        Save the DataFrame with gender inference results to a CSV file.
        """
        if self.givennames_df is None:
            raise ValueError("Given names DataFrame is not loaded.")

        if GENDER_COLUMN not in self.givennames_df.columns:
            raise ValueError(f"The DataFrame must contain a '{GENDER_COLUMN}' column.")

        ios.to_csv(self.givennames_df.set_index(GIVENNAME_COLUMN), filepath, index=False)

