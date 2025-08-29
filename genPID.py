# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 15:43:05 2025

@author: JTM (from Miles Mahon)
"""

import os
import hashlib

def genPID(seed):
    if seed[0].lower() == 'e' and seed[1].lower() == '_':
        pid = seed[2:]
        return pid
    p = os.getenv('bci_encode_pass')
    if not p:
        p = input('input the conversion string: ')
        os.environ['bci_encode_pass'] = p
    encodedIdLength = 6
    modulator = hash(seed + p)
    pid = modulator[:encodedIdLength]
    return pid

def hash(inp):
    inp = inp.encode('utf-8')
    x = hashlib.sha256()
    x.update(inp)
    h = x.digest()
    h = h.hex()
    return h


if __name__ == '__main__':
    seed= input('Enter the subject ID seed: ')
    print('The PID is: ' + genPID(seed))