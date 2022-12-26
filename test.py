import os
print(__file__)
folder = os.path.dirname(__file__)
father_folder = os.path.dirname(os.path.dirname(__file__))
print(father_folder)
print(os.path.abspath(r".."))