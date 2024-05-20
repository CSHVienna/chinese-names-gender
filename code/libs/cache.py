
import os
import json
import gzip

class DiskCache:

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def get_or_generate(self, id_str, fnc):
        fn = f"{os.path.join(self.cache_dir, str(id_str))}.gz"
        if os.path.exists(fn):
            print("loading cache...")
            # Reading compressed JSON from a file
            with gzip.open(fn, 'rt') as file:  # 'rt' mode for text reading
                obj = json.load(file)
        else:
            print("writing cache...")
            # Writing compressed JSON to a file
            obj = fnc()
            if obj is not None:
                with gzip.open(fn, 'wt') as file:  # 'wt' mode for text writing
                    json.dump(obj, file, indent=2)
        return obj
