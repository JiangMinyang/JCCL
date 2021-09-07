import configparser
import os


class Config(object):
    def __init__(self):
        basepath = os.path.dirname(__file__)
        self.filepath = 'config.ini'
        self._config = configparser.ConfigParser()
        self._config.read(self.filepath)
        self.scope = 'Train'

    def set_scope(self, scope):
    	self.scope = scope
    	
    def getfloat(self, name):
        return self._config.getfloat(self.scope, name)

    def getint(self, name):
        return self._config.getint(self.scope, name)

    def get(self, name):
        return self._config.get(self.scope, name)

    def getbool(self, name):
    	return self._config.getboolean(self.scope, name)

config = Config()