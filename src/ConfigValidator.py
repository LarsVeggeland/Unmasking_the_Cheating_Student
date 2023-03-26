# ---------- Imported libraries --------



# ---------- Util functions ----------

def vaildate_dataset_fields(dataset : str, word_cap : int, file_partitions : bool, balanced_classes : bool) -> list:
    """
    Checks whether all the datset related fields in the provided config file have appropriate values
    """
    errors = []
    try:
        assert(len(dataset) > 0)
    except AssertionError:
        errors.append(f"No datatset specified")
    
    try:
        with open(dataset, "r") as file:
            pass
    except FileNotFoundError:
        errors.append(f"The provided dataset {dataset} could not be found in the filesystem")

    try:
        assert(dataset[-4:] == ".csv")
    except AssertionError:
        errors.append(f"The provided dataset {dataset} is not a csv file")

    try:
        assert(word_cap is None or isinstance(word_cap, int))
    except AssertionError:
        errors.append(f"The word cap must be null or an integer not {type(word_cap)}")

    try:
        if isinstance(word_cap, int):
            assert(word_cap > 0)
    except AssertionError:
        errors.append(f"If not null the word cap must be a positive integer not {word_cap}")

    try:
        assert(isinstance(file_partitions, bool))
    except AssertionError:
        errors.append(f"The field 'file_partitions' must be defined as either true or false not {file_partitions} of type {type(file_partitions)}")
    
    try:
        if not isinstance(word_cap, int):
            assert(file_partitions == False)
    except AssertionError:
        errors.append(f"Cannot set 'file_partitions' to true when when no word cap is enforced")
    
    try:
        assert(isinstance(balanced_classes, bool))
    except AssertionError:
        errors.append(f"The field 'balanced_classes' must be defined as either true or false not {balanced_classes} of type {type(balanced_classes)}")

    return errors


def validate_chunk_fields(conf : dict) -> list:
    """
    Checks whether all the chunk related fields are correctly defined
    """
    errors = []
    try:
        size = conf["size"]
        assert(size > 0)
    except AssertionError:
        errors.append(f"Chunk size must be a positive integer not {size}")

    try:
        type = conf['type']
        assert(type in ["words", "ngrams", "pos_tags", "lex_pos"])

        if type == "ngrams":
            try:
                ngram_size = conf["ngram_size"]
                assert(ngram_size > 0 and isinstance(ngram_size, int))
            except AssertionError:
                errors.append(f"ngram size must be a positive integer not {self.ngram_size}")
            except KeyError:
                    errors.append("Chunk type set to ngrams but ngram size is not defined in configuration file")
    except AssertionError:
        errors.append(f"The provided chunk type {type} does not exist")

    try:
        method = conf["method"]
        assert(method in ["original", "bootstrap", "sliding_window"])
    except AssertionError:
        errors.append(f"The provided chunk method {method} does not exits")
    
    return errors


def validate_feature_extractor_fields(conf : dict) -> list:
    # TODO
    pass


def validate_curve_related_fields(conf : dict) -> list:
    # TODO
    pass


# ---------- Configuration file validation ---------

def validate_config_file(filename : str) -> None:
    # TODO
    pass