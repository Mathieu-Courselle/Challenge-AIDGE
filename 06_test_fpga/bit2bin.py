#!/usr/bin/env python3
"""Convertit un .bit Vivado en .bin 'byte swapped' pour le FPGA Manager Zynq-7000.
Usage: python3 bit2bin.py fpga.bit fpga_swapped.bin
"""
import struct, sys

src, dst = sys.argv[1], sys.argv[2]
d = open(src, 'rb').read()

# en-tete .bit : 2 octets longueur + champ + 2 octets longueur + 1, puis sections a,b,c,d,e
n = struct.unpack('>H', d[:2])[0]
p = 2 + n
p += 2                      # 0x0001
for key in b'abcd':
    assert d[p] == key, 'en-tete .bit inattendu (section %r)' % chr(key)
    l = struct.unpack('>H', d[p+1:p+3])[0]
    p += 3 + l
assert d[p] == ord('e'), 'section e introuvable'
L = struct.unpack('>I', d[p+1:p+5])[0]
raw = d[p+5:p+5+L]
assert len(raw) == L and L % 4 == 0, 'taille de donnees incorrecte'
assert b'\xaa\x99\x55\x66' in raw[:200], 'mot de synchronisation introuvable'

out = bytearray(L)
for i in range(0, L, 4):
    out[i:i+4] = raw[i:i+4][::-1]
assert b'\x66\x55\x99\xaa' in bytes(out[:200])
open(dst, 'wb').write(out)
print('OK : %d octets ecrits dans %s' % (L, dst))
