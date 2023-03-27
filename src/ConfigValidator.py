# ---------- Imported libraries --------

from json import loads


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
                errors.append(f"ngram size must be a positive integer not {ngram_size}")
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
    """
    Checks the validity of the fields related to feature extraction
    """
    errors = []
    try:
        type = conf["type"]
        assert(type in ["words", "ngrams", "pos_tags", "lex_pos"])
    except AssertionError:
        errors.append(f"The provided feature type {type} does not exist")
    except KeyError:
        errors.append(f'The mandatory field "type" has been omitted from the configuration file')

    try:
        normalized = conf["normalized"]
        assert(isinstance(normalized, bool))
    except AssertionError:
        errors.append(f"The field normalized must be a boolean not {type(normalized)}")
    except KeyError:
        errors.append(f'The mandatory field "normalized" has been omitted from the configuration file')
            
    try:
        number = conf["number"]
        assert(isinstance(number, int) and number > 0)
    except AssertionError:
        errors.append(f"The field number must be a positive integer not {number}")
    except KeyError:
        errors.append(f'The mandatory field "number" has been omitted from the configuration file')


def validate_curve_related_fields(build : bool, eliminate : int, save : str, load : str, C_cons : float, C_class : float, kernel : str) -> list:
    errors = []

    try:
        assert(isinstance(build, bool))
    except AssertionError:
        errors.append(f"The field 'file_partitions' must be defined as either true or false not {build} of type {type(build)}")

    try:
        assert(isinstance(eliminate, int))
    except AssertionError:
        errors.append(f"The field 'elimnate' must be an integer, not {type(eliminate)}")

    try:
        if isinstance(eliminate, int):
            assert(eliminate > 0)
    except AssertionError:
        errors.append(f"The field 'eliminate_features' must be a positive integer not {eliminate}. Cannot remove a negative number of features...")
    
    try:
        assert(isinstance(build, bool))
        if build:
            assert(isinstance(save, str), f"The field specifying the target file for saving the created author curves must be a string not {type(save)}")
            assert(len(save) > 4, "The target file for saving the created author curves is not defined")
            assert(save[-4:] == ".csv", f"The provided dataset {save} is not a csv file")
            assert(isinstance(load, str), f"The field specifying the source file for loading the author curves must be a string not {type(save)}")
            assert(len(load) > 4, "The source file for loading the author curves is not defined")
            assert(save == load, f"Inconsistency between the author curves created and the ones used during testing. Not loading the created author curves {save} =/= {load}")
    except AssertionError as e:
        errors.append(e)

    try:
        assert(isinstance(C_cons, float), f"The field 'C_parameter_curve_construction' must be a float not {type(C_cons)}")
        # C can be larger than 1, but that would be weird...
        assert(C_cons <= 1 and C_cons >= 0, f"The field 'C_parameter_curve_construction' must be in the range [0, 1] not {C_cons}")    
    except AssertionError as e:
        errors.append(e)

    try:
        assert(isinstance(C_class, float), f"The field 'C_parameter_curve_classification' must be a float not {type(C_cons)}")
        # C can be larger than 1, but that would be weird...
        assert(C_class <= 1 and C_class >= 0, f"The field 'C_parameter_curve_classification' must be in the range [0, 1] not {C_class}")    
    except AssertionError as e:
        errors.append(e)

    try:
        assert(isinstance(kernel, str, f"The field 'kernel_type_curve_classification' must be a string, not {type(kernel)}"))
        allowed =  ['linear', "rbf", "poly", "sigmoid", "precomputed"]
        assert(kernel in allowed, f"The field 'kernel_type_curve_classification' must be among {' '.join(allowed)}, not {kernel}")
    except AssertionError as e:
        errors.append(e)
    
    return errors



# ---------- Configuration file validation ---------

def validate_config_file(filename : str) -> bool:
    with open(filename, "r") as file:
        config_file : dict = loads(file)
    
    dataset = config_file["dataset"]
    word_cap = config_file["word_cap"]
    file_partitions = config_file["file_partitions"]
    balanced_classes = config_file["balanced_classes"]

    chunk_config = config_file["chunk_config"]
    feature_config = config_file["feature_config"]

    build = config_file["build_author_curves"]
    eliminate = config_file["eliminate"]
    save = config_file["save_author_curves"]
    load = config_file["load_author_curves"]
    C_cons = config_file["C_parameter_curve_construction"]
    C_class = config_file["C_parameter_curve_classification"]
    kernel = config_file["kernel_type_curve_classification"]

    errors = []

    # Test dataset related fields
    errors.append(vaildate_dataset_fields(dataset, word_cap, file_partitions, balanced_classes))

    # Test chunk related fields
    errors.append(validate_chunk_fields(chunk_config))

    # Test feature extraction related fields
    errors.append(validate_feature_extractor_fields(feature_config))

    # Test curve related fields
    errors.append(validate_curve_related_fields(build, eliminate, save, load, C_cons, C_class, kernel))

    if not len(errors):
        return True

    error_msg = "\n".join(errors)

    print(f"There are issues with the provided configuration file {filename}:\n{error_msg}")
    return False