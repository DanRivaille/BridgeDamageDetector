import argparse
import os


class ParserArguments:
    """
    A class for parsing command line arguments related to a model.
    """

    def __init__(self):
        """
        Initializes an instance of ParserArguments.
        """
        self.__parser = argparse.ArgumentParser()
        self.__create_arguments()
        self.__args = self.__parser.parse_args()

    def __create_arguments(self):
        """
        Defines command line arguments and adds them to the argument parser.
        """
        self.__parser.add_argument("--id", help="Model identifier", required=True)
        self.__parser.add_argument("--config", help="Configuration file", required=True)
        self.__parser.add_argument("-s", "--save", help="Save the trained model and its results", default=False,
                                   action='store_true')
        self.__parser.add_argument("-t", "--test", help="Run the test process", default=False, action='store_true')
        self.__parser.add_argument("-o", "--output-folder", nargs="?", help="The folder name to save the runs.",
                                   default="example")

    @property
    def model_id(self) -> str:
        """
        Gets the identifier of the model used.
        """
        return self.__args.id

    @property
    def config_filename(self) -> str:
        """
        Gets the configuration file name of the model used.
        """
        return os.path.split(self.__args.config)[1]

    @property
    def save(self) -> bool:
        """
        Gets the save flag. If true, indicates that the trained model and its results will be saved.
        """
        return self.__args.save

    @property
    def is_test(self) -> bool:
        """
        Gets the test flag. If true, indicates that the testing process will be run.
        """
        return self.__args.test

    @property
    def folder_name(self) -> str:
        """
        Gets the folder name to save the runs.
        """
        return self.__args.output_folder
