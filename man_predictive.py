import pandas as pd
from sklearn import tree
import openpyxl


class PredManClass:
    def __init__(self):
        self.vib_foot = pd.read_excel('VibrationFootprints.xlsx', sheet_name='Foglio1')

    def some_stats(self):
        v = self.vib_foot['classe'].value_counts()
        print('N° sample x classe:')
        print(v)


if __name__ == "__main__":
    predManObj = PredManClass()
    predManObj.some_stats()
