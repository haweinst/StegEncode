usinglzma = False
import numpy as np
from PIL import Image

with open('widthheight.txt') as f:
    listofthings = f.read().split(',')
    w = listofthings[0]
    h = listofthings[1]
f.close()


# define bitwise xor for integers greater than maxsize
def bitxor(a, b):
    # pad with zeroes
    s = ''
    maxlen = max(len(a), len(b))
    a = a.zfill(maxlen)
    b = b.zfill(maxlen)
    for i in range(maxlen):
        if (a[i] == b[i]):
            s += '0'
        else:
            s += '1'
    return s.zfill(maxlen)


# written by chatgpt

def png_to_binary(file_path):
    """
    Read a PNG file and convert it into a binary string.

    Parameters:
        file_path (str): The path to the PNG file.

    Returns:
        str: Binary string representing the contents of the PNG file.
    """
    try:
        # Open the image using PIL
        img = Image.open(file_path)

        # Convert the image to grayscale and then to a numpy array of uint8 values
        img_gray = img.convert('L')
        img_array = np.array(img_gray, dtype=np.uint8)

        # Determine the dimensions of the original image
        height, width = img_array.shape

        # Convert the numpy array to a binary string
        binary_string = ''.join(format(byte, '08b') for byte in img_array.flatten())
        return binary_string, width, height

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


# end chatgpt code

a, w, h = png_to_binary('resized.png')

with open('binarystringmeasured.txt', 'r') as f:
    ppp = f.read()

measuredbits = ppp  # put whatever you measured here. should be a binary message
print(len(measuredbits))
import lzma
from lzma import FORMAT_XZ
import numpy as np
import scipy as sp

lengthofcodeword = 2 ** 3

loh = 5
lenb = 1484
bl = lenb - loh


# prefix = '\xfd7zXZ'
# lom = 774 #length of message

def counterr(q, v):
    print('----')
    print(len(q))
    print(len(v))
    count = 0
    for i in range(0, len(q)):
        if q[i] != v[i]:
            count = count + 1
    return count


# a = '01010001001011110011101'
z = a

if (usinglzma):
    b = lzma.compress(a.encode(), format=FORMAT_XZ, preset=1)
    print(b[0:20])
    print(b[0:loh])
    bl = len(b) - loh
    print(bl)
    z = bin(int.from_bytes(b[loh:], byteorder='big'))

    z = z[2:]  # eliminate b'

messagedecoded = ""

numreps = 2 ** 3


# maxweight = int(numreps/2)
# def lenstabs(nr, mw):
#     u = 0
#     for i in range(0, mw+1):
#         u+=sp.special.comb(nr, i)
#     return int(u)

# mn = 0
# s=''
# stabilizers = [None]*(lenstabs(numreps, maxweight))
# for i in range(0, maxweight):
#     for k in range(0, numreps):
#         if i==0:
#             for j in range(0, k):
#                 s = '0'*j
#                 s+='1'
#                 s+='0'*(k-j-1)
#                 s+='1'
#                 s+='0'*(numreps-k-1)
#                 stabilizers[mn]=s
#                 mn +=1 
#         else:
#             s='0'*k
#             s+='1'
#             s+='0'*(numreps-k-1)
#             stabilizers[mn]=s
#             mn=mn+1

# stabilizers[-1]='0'*numreps
# if(not usinglzma):
#     stbs = ['0']*(2*len(stabilizers))
#     for i in range(0, len(stabilizers)):
#         stbs[i] = stabilizers[i]
#         stbs[len(stabilizers)+i] = np.binary_repr(np.bitwise_xor(int('1'*lengthofcodeword, 2), int(stabilizers[i], 2)), width=lengthofcodeword)
#     stabilizers = stbs
# print(stabilizers)
# define encoding and decoding methods for code
def encode(msg):
    result = ""
    for i in msg:
        for z in range(0, numreps):
            result += i
    return result


def decode(msg):
    s = ""
    for i in range(0, int(len(msg) / lengthofcodeword)):
        s += decodeCodeword(msg[lengthofcodeword * i:lengthofcodeword * (i + 1)])
    return s


def decodeCodeword(m):
    if (m.count('0') > m.count('1')):
        return '0'
    return '1'


# end of 3bit code definition


# step 3: choose a random correctable error of the code to multiply by and compute encoded message
# nstabs = len(stabilizers)
with open('secretkey.txt', 'r') as f:
    rawstring = f.read()
encodingkey = rawstring.split(',')[0:-1]

# step 8: reverse the key
for i in range(0, len(z)):
    encbloc = bitxor(measuredbits[i * lengthofcodeword:(i + 1) * lengthofcodeword], encodingkey[i])
    # print(len(encbloc))
    messagedecoded += encbloc
# step 9: decode the message
messagedecoded = decode(messagedecoded)
print(len(z))
print(len(messagedecoded))
print(counterr(messagedecoded, z))

c = messagedecoded

if (usinglzma):
    y = int(messagedecoded, 2).to_bytes(bl, byteorder='big')
    hn = "b'" + str(b)[7 - loh:10] + str(y)[2:]
    print(hn[0:40])
    s = eval(hn)
    c = lzma.decompress(s, format=FORMAT_XZ)
    c = c.decode()


def binary_to_png(binary_string, width, height, output_file):
    """
    Convert a binary string into a PNG file.

    Parameters:
        binary_string (str): Binary string representing the contents of the PNG file.
        width (int): Width of the original image.
        height (int): Height of the original image.
        output_file (str): The path to save the output PNG file.
    """
    try:
        # Convert binary string back to a numpy array of uint8 values
        img_array = np.array([int(binary_string[i:i + 8], 2) for i in range(0, len(binary_string), 8)], dtype=np.uint8)

        # Reshape the array to the original image dimensions
        img_array = img_array.reshape((height, width))

        # Create an image from the numpy array
        img = Image.fromarray(img_array)

        # Save the image to the output file
        img.save(output_file)
        print("Image saved successfully!")

    except Exception as e:
        print(f"Error: {e}")


print('the number of errors is ' + str(counterr(a, c)))

with open('binaryoutputstringihave.txt', 'w') as g:
    g.write(c)


# below code only written by chatgpt

def binary_to_ascii(binary_string):
    """
    Convert a binary string to ASCII characters.
    
    Parameters:
    - binary_string (str): The binary string to convert.
    
    Returns:
    - str: The ASCII representation of the binary string.
    """
    # Split the binary string into 8-bit chunks
    chunks = [binary_string[i:i + 8] for i in range(0, len(binary_string), 8)]

    # Convert each 8-bit chunk to its corresponding ASCII character
    ascii_chars = [chr(int(chunk, 2)) for chunk in chunks]

    # Concatenate the ASCII characters to form the result string
    ascii_string = ''.join(ascii_chars)

    return ascii_string


# end of chatgpt code


import os
import io
import PIL.Image as Image

# print('###')
# print(c)
binary_to_png(c, w, h, 'converted_trojan.png')
