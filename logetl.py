import re
import pandas as pd

class LogETL():
    def __init__(self):
        self.parse_string = '([(\d\.)]+) - - \[(.*?)\] "(.*?)" (\d+) (\d*)'

    def extract(self,filename):
        with open(filename) as f:
            for line in f:
                re.match(self.parse_string, line).groups()