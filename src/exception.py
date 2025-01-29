import sys
from src.logger import logging


def error_message_detail(error_message, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    custom_error = f"Error occured in python script file name {file_name} \
at line no {exc_tb.tb_lineno}. \
Error message: {str(error_message)}"
    return custom_error


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        logging.info(self.error_message)

    def __str__(self):
        return self.error_message
