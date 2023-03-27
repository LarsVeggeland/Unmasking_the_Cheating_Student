# ---------- Imported libraries --------

from json import load as json_load
from itertools import chain


# ---------- Util functions ----------

def vaildate_dataset_fields(dataset, word_cap, file_partitions, balanced_classes) -> list:
    """
    Checks whether all the datset related fields in the provided config file have appropriate values
    """
    errors = []
    try:
        assert isinstance(dataset, str), f"No datatset specified"
        assert len(dataset) > 0, f"No datatset specified"
        assert dataset[-4:] == ".csv", f"The provided dataset {dataset} is not a csv file"
        with open(dataset, "r") as file:
            pass
    except AssertionError as e:
        errors.append(str(e))
    except IndexError:
        errors.append(f"The provided dataset {dataset} is not a csv file")
    except FileNotFoundError:
        errors.append(f"The provided dataset {dataset} could not be found in the filesystem")

    try:
        assert word_cap is None or isinstance(word_cap, int), f"The word cap must be null or an integer not {type(word_cap)}"
        if isinstance(word_cap, int):
            assert word_cap > 0, f"If not null the word cap must be a positive integer not {word_cap}"
    except AssertionError as e:
        errors.append(str(e))

    try:
        assert isinstance(file_partitions, bool), f"The field 'file_partitions' must be defined as either true or false not {file_partitions} of type {type(file_partitions)}"
        if not isinstance(word_cap, int):
            assert file_partitions == False, "Cannot set 'file_partitions' to true when when no word cap is enforced"
    except AssertionError as e:
        errors.append(str(e))
    
    try:
        assert isinstance(balanced_classes, bool)
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
        assert isinstance(size, int)
        assert size > 0
    except AssertionError:
        errors.append(f"Chunk size must be a positive integer not {size}")

    try:
        type = conf['type']
        types = ["words", "ngrams", "pos_tags", "lex_pos"]
        assert type in types, f"The provided chunk type {type} does not exist. Must be among {' '.join(types)}"

        if type == "ngrams":
            ngram_size = conf["ngram_size"]
            assert isinstance(ngram_size, int), f"ngram size must be a positive integer not {ngram_size}"
            assert ngram_size > 0, f"ngram size must be a positive integer not {ngram_size}"
    except AssertionError as e:
        errors.append(str(e))
    except KeyError:
        errors.append("Chunk type set to ngrams but the field 'ngram_size' is not defined in configuration file")

    try:
        method = conf["method"]
        methods = ["original", "bootstrap", "sliding_window"]
        assert method in methods
    except AssertionError:
        errors.append(f"The provided chunk method {method} does not exits. Must be among {', '.join(methods)}")
    
    return errors


def validate_feature_extractor_fields(conf : dict) -> list:
    """
    Checks the validity of the fields related to feature extraction
    """
    errors = []
    try:
        t = conf["type"]
        types = ["words", "ngrams", "pos_tags", "lex_pos"]
        assert t in types, f"The provided feature type {type} does not exist. Must be among {', '.join(types)}"
    except AssertionError as e:
        errors.append(str(e))
    except KeyError:
        errors.append(f"The mandatory field 'type' in 'feature_config' has been omitted from the configuration file")

    try:
        normalized = conf["normalized"]
        assert isinstance(normalized, bool)
    except AssertionError:
        errors.append(f"The field normalized must be a boolean not {type(normalized)}")
    except KeyError:
        errors.append(f'The mandatory field "normalized" has been omitted from the configuration file')
            
    try:
        number = conf["number"]
        assert isinstance(number, int)
        assert number > 0 
    except AssertionError:
        errors.append(f"The field 'number' in 'feature_config' must be a positive integer not {number}")
    except KeyError:
        errors.append(f"The mandatory field 'number' in 'feature_config' has been omitted from the configuration file")


def validate_curve_related_fields(build, eliminate, save, load, C_cons, C_class, kernel) -> list:
    errors = []

    try:
        assert isinstance(build, bool) 
    except AssertionError:
        errors.append(f"The field 'file_partitions' must be defined as either true or false not {build} of type {type(build)}")

    try:
        assert isinstance(eliminate, int) 
    except AssertionError:
        errors.append(f"The field 'elimnate' must be an integer, not {type(eliminate)}")

    try:
        if isinstance(eliminate, int):
            assert eliminate > 0 
    except AssertionError:
        errors.append(f"The field 'eliminate_features' must be a positive integer not {eliminate}. Cannot remove a negative number of features...")
    
    try:
        assert isinstance(build, bool)
    except AssertionError:
        errors.append(f"The field 'build_author_curves' must be a boolean, not {type(build)}")

    try:
        assert isinstance(load, str), f"The field specifying the source file for loading the author curves must be a string not {type(save)}"
        assert len(load) > 4, "The source file for loading the author curves is not defined"
        assert load[-4:] == ".csv", f"The provided source file for the loaded author curves {load} is not a csv file" 
        if not build:
            with open(load, "r") as file:
                pass
    except AssertionError as e:
        errors.append(str(e))
    except FileNotFoundError:
        errors.append(f"The config file specifies that no author curves are to be constructed, however, the target file for loading the preconstructed author curves {load} cannot be found.")

    try:
        if build:
            assert isinstance(save, str), f"The field specifying the target file for saving the created author curves must be a string not {type(save)}"
            assert len(save) > 4, "The target file for saving the created author curves is not defined"
            assert save[-4:] == ".csv", f"The provided target file for saving the author curves {save} is not a csv file"
            assert save == load, f"Inconsistency between the author curves created and the ones used during testing. Not loading the created author curves {save} =/= {load}"
    except AssertionError as e:
        errors.append(str(e))

    try:
        assert isinstance(C_cons, float), f"The field 'C_parameter_curve_construction' must be a float not {type(C_cons)}"
        # C can be larger than 1, but that would be weird...
        assert C_cons <= 1 and C_cons >= 0, f"The field 'C_parameter_curve_construction' must be in the range [0, 1] not {C_cons}"   
    except AssertionError as e:
        errors.append(str(e))

    try:
        assert isinstance(C_class, float), f"The field 'C_parameter_curve_classification' must be a float not {type(C_cons)}"
        # C can be larger than 1, but that would be weird...
        assert C_class <= 1 and C_class >= 0, f"The field 'C_parameter_curve_classification' must be in the range [0, 1] not {C_class}"    
    except AssertionError as e:
        errors.append(str(e))

    try:
        assert isinstance(kernel, str), f"The field 'kernel_type_curve_classification' must be a string, not {type(kernel)}"
        allowed =  ['linear', "rbf", "poly", "sigmoid", "precomputed"]
        assert kernel in allowed, f"The field 'kernel_type_curve_classification' must be among {' '.join(allowed)}, not {kernel}"
    except AssertionError as e:
        errors.append(str(e))
    
    return errors



# ---------- Configuration file validation ---------

def validate_config_file(filename : str) -> bool:
    with open(filename, "r") as file:
        config_file : dict = json_load(file)
    
    dataset = config_file["dataset"]
    word_cap = config_file["word_cap"]
    file_partitions = config_file["file_partitions"]
    balanced_classes = config_file["balanced_classes"]

    chunk_config = config_file["chunk_config"]
    feature_config = config_file["feature_config"]

    build = config_file["build_author_curves"]
    eliminate = config_file["features_eliminated" ]
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
    
    error_msg = ""

    for i in range(4):
        if errors[i] is None:
            continue
        for j in range(len(errors[i])):
            error = errors[i][j]
            if error is not None:
                error_msg += "[!] - " + error + "\n"
    

    if not len(error_msg):
        return True

    print(f"There are issues with the provided configuration file {filename}:\n{error_msg}")
    return False


file = "conf/test.json"

print(validate_config_file(file))