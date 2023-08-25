import logging

URL = "utilities/"

class CustomLogging:
    def __init__(self, name, level=logging.DEBUG, filename=URL+"CustomLogging.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s')

        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler) # Se agrega handler para stream

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
