# ---------- Imported libraries ----------

from datetime import datetime



# ---------- Util functions ----------

def print_progressbar(current_position : int, length : int, granularity : int = 100) -> None:
    """
    Prints a progressbar to the terminal
    """
    progress = int(((current_position+1)/length)*100)
    line = "="*progress + "-"*(granularity-progress)
    progress_str = f"[{line}] {progress} % ({current_position+1}/{length})"
    if current_position < length-1: print(progress_str, end='\r')
    else: print(progress_str)


def get_time() -> str:
    """
    Gets the current time
    """
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")