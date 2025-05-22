FileNotFoundError                         Traceback (most recent call last)
<ipython-input-4-12a3de4ffaef> in download_dataset()
     75         else:
---> 76             raise FileNotFoundError("Kaggle API credentials not found. Place kaggle.json in working directory or set KAGGLE_USERNAME/KAGGLE_KEY.")
     77 

FileNotFoundError: Kaggle API credentials not found. Place kaggle.json in working directory or set KAGGLE_USERNAME/KAGGLE_KEY.

During handling of the above exception, another exception occurred:

SystemExit                                Traceback (most recent call last)
    [... skipping hidden 1 frame]

10 frames
SystemExit: 1

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
    [... skipping hidden 1 frame]

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in find_recursion(etype, value, records)
    380     # first frame (from in to out) that looks different.
    381     if not is_recursion_error(etype, value, records):
--> 382         return len(records), 0
    383 
    384     # Select filename, lineno, func_name to track frames with

TypeError: object of type 'NoneType' has no len()
