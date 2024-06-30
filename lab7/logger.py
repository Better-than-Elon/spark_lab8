import logging
import os
import sys

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = os.path.join(os.getcwd(), "logfile.log")


# noinspection PyMethodMayBeStatic
class Logger:
    """
        Class for logging behaviour of data exporting - object of ExportingTool class
    """

    def __init__(
            self,
            show: bool,
            log_to_file: bool = False,
    ) -> None:
        """
            Re-defined __init__ method which sets show parametr

        Args:
            show (bool): if set all logs will be shown in terminal
            log_to_file (bool): if set all logs will be stored to LOG_FILE
        """
        self.show = show
        self.log_to_file = log_to_file

    def get_console_handler(self) -> logging.StreamHandler:
        """
            Class method the aim of which is getting a console handler to show logs on terminal

        Returns:
            logging.StreamHandler: handler object for streaming output through terminal
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(FORMATTER)
        return console_handler

    def get_file_handler(self) -> logging.FileHandler:
        """
            Class method the aim of which is getting a file handler to write logs in file LOG_FILE

        Returns:
            logging.FileHandler: handler object for streaming output through std::filestream
        """
        file_handler = logging.FileHandler(LOG_FILE, mode='w')
        file_handler.setFormatter(FORMATTER)
        return file_handler

    def get_logger(self, logger_name: str) -> logging.Logger:
        """
            Class method which creates logger with certain name

        Args:
            logger_name (str): name for logger

        Returns:
            logger: object of Logger class
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        if self.show:
            logger.addHandler(self.get_console_handler())
        if self.log_to_file:
            logger.addHandler(self.get_file_handler())
        logger.propagate = False
        return logger