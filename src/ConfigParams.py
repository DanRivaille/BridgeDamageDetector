from json import load, dump


class ConfigParams:
    """
    A class for loading and saving configuration parameters related to a model.
    """

    def __init__(self, config_dict: dict, original_filepath: str):
        """
        Initializes an instance of ConfigParams.
        @param config_dict The dictionary containing configuration parameters.
        @param original_filepath The file path containing those parameters.
        """
        self.params = config_dict
        self.original_filepath = original_filepath
        self.global_variables: dict = config_dict.get("global_variables")

    def save(self, overwrite_original: bool = True, new_path: str = "") -> None:
        """
        Saves the configuration parameters to a file.
        @param overwrite_original A flag that, if True overwrites the original file path with the configuration
        parameters; otherwise, they are saved to a new path. Default is True.
        @param new_path The path to save the configuration parameters if not overwriting the original. Default is "".
        """
        if overwrite_original:
            path_to_save = self.original_filepath
        else:
            path_to_save = new_path

        with open(path_to_save, 'w') as config_file:
            dump(self.params, config_file)

    @staticmethod
    def load(filepath: str):
        """
        Loads the configurations parameters from a file.
        @param filepath The path to the configuration file.
        """
        with open(filepath, 'r') as config_file:
            config_params = load(config_file)

        return ConfigParams(config_params, filepath)

    def get_params(self, key_param: str):
        """
        Returns the specific group of validated configuration parameters from the dictionary for the given key (validates depending on whether they are global variables or not).
        @param key_param The key of the parameters set to retrieve.
        """
        params = self.params.get(key_param, None)
        return self.__validate_param(params)

    def __validate_param(self, param_to_validate):
        """
        Validates that the parameter and its sub-parameters use their values from global variables if they have one.
        @param param_to_validate The parameter or group of parameters to be validated.
        """
        if isinstance(param_to_validate, list):
            params = []
            for param in param_to_validate:
                params.append(self.__validate_param(param))
            return params

        elif isinstance(param_to_validate, dict):
            params = {}
            for key in param_to_validate:
                params[key] = self.__validate_param(param_to_validate[key])
            return params

        elif isinstance(param_to_validate, str):
            value_in_global_variables = self.global_variables.get(param_to_validate)
            if value_in_global_variables is not None:
                return value_in_global_variables

        return param_to_validate
