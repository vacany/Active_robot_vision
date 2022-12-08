import socket
import os

if socket.gethostname() == 'Patrik':
    argoverse2 = '/home/patrik/patrik_data/argoverse2/'

# RCI
else:
    argoverse2 = f'{os.path.expanduser("~")}/data/argoverse2/'
