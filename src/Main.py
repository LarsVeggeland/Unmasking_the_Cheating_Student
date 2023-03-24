# ---------- Imported libraries ----------

from os import listdir, path
from Pipeline import Pipeline



# ---------- Util functions ----------

def run_config(filename : str) -> None:
    """
    Configures and runs an instance of the Pipeline
    based on the provided config file
    """
    Pipeline(settings_file=filename)


def run_all_config_files_in_dir(dirname : str) -> None:
    """
    Configures and runs a Pipeline instance
    for each of the config files in the 
    provided directory
    """
    config_files = listdir(dirname)
    for file in config_files:
        if file[-6:] != ".json":
            pass
        print(f"Runnig pipeline configured by {file}")
        run_config(dirname + "/" + file)

    

# ---------- Main ----------

def __main__(source : str) -> None:
    """
    Main function
    """
    if path.isdir(source):
        run_all_config_files_in_dir(source)
    elif path.isfile(source):
        run_config(source)
    else:
        print(f"\nERROR: {source} is neither a file nor directory")
        exit(1)
    

if __name__ == "__main__":
    __main__("conf\PreliminaryTesting")