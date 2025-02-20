usinglzma = False
# add errors if desired
errrate = 0.5
# finite set of states
numsamples = 1000
samplezeroes = [0] * int(numsamples / 2)
sampleones = [0] * int(numsamples / 2)
angzeroes = [0] * int(numsamples / 2)
angones = [0] * int(numsamples / 2)

import time

timestart = time.time()


def makeRM(r, m):
    # print(str(r)+','+str(m))
    if r == 0:
        return np.array([1] * (2 ** m))
    if m == r:
        return np.eye(2 ** m)

    else:
        a = makeRM(r - 1, m - 1)
        return np.block([
            [makeRM(r, m - 1), makeRM(r, m - 1)],
            [np.zeros(a.shape), a]
        ])


def genRayleighs(enbar):
    rays = np.random.rayleigh(scale=np.sqrt(enbar / 2), size=1000 * numsamples)
    for i in range(0, len(samplezeroes)):
        samplezeroes[i] = np.quantile(rays, (2 * i + 1) / (2 * numsamples))
        angzeroes[i] = 2 * np.pi * (2 * i + 1) / (2 * numsamples)
    for i in range(len(samplezeroes), numsamples):
        sampleones[i - len(samplezeroes)] = np.quantile(rays, (2 * i + 1) / (2 * numsamples))
        angones[i - len(samplezeroes)] = 2 * np.pi * (2 * i + 1) / (2 * numsamples)


# def drawRayleigh(bit, enbar):
#     rad = np.random.rayleigh(scale =  np.sqrt(enbar/2))
#     ang = np.random.uniform(0, 2*np.pi)
#     if bit ==0:    

#         while rad > np.sqrt(enbar*np.log(2)):
#             rad = np.random.rayleigh(scale =  np.sqrt(enbar/2))
#     else:
#         while rad < np.sqrt(enbar*np.log(2)):
#             rad =  np.random.rayleigh(scale =  np.sqrt(enbar/2))
#     return [rad, ang]

def drawRayleigh(bit, enbar):
    indr = np.random.randint(0, len(sampleones))
    inda = np.random.randint(0, len(sampleones))
    if bit == 0:
        return [samplezeroes[indr], angzeroes[inda]]
    else:
        return [sampleones[indr], angones[inda]]


import numpy as np
from PIL import Image


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


# #above code only written by chatgpt
a, w, h = png_to_binary('resized.png')
with open('widthheight.txt', 'w') as f:
    f.write(str(w) + ',' + str(h))
f.close()

import lzma
from lzma import FORMAT_XZ
import numpy as np
import scipy as sp

z = a
# a='01010001001011110011101' #test
if (usinglzma):
    loh = 5  # length of header
    b = lzma.compress(a.encode(), format=FORMAT_XZ, preset=1)
    bl = len(b) - loh  # 1479
    z = bin(int.from_bytes(b[loh:], byteorder='big'))

    z = z[2:]  # eliminate b'

# step 1: choose a message and a channel
nbar = 1.26  # taken from figure 4 in the paper
genRayleighs(nbar)  # generate rayleigh samples
messageog = z
print(len(messageog))
message = messageog
# print('len = '+str(len(z)))

# step 2: choose an error correcting code suitable for the error rate p_err
# perrhomo = estimate_perr_homodyne(nbar)
# perrhetero = estimate_perr_heterodyne(nbar)
# print(perrhomo)


# start of RM(r,m) definition
order = 0
m = 5
blocksize = 2 ** m
print(blocksize)
numreps = blocksize
lengthofcodeword = blocksize


def compute_error_probability_repetition(prob, nn):
    # sum over all weights of errors with appropriate probabilities
    zed = 0
    for i in range(int(nn / 2) + 1, nn + 1):
        zed += sp.special.comb(nn, int(nn / 2) + 1) * prob ** (int(nn / 2) + 1) * (1 - prob) ** (int(nn / 2))
    return zed


# print('the error probability for a given codeword is approximately '+str(compute_error_probability_repetition(perrhetero, numreps)))

# construct random bit strings of length lengthofcodeword
# maxweight = int(lengthofcodeword/2)
# def lenstabs(nr, mw):
#     u = 0
#     for i in range(0, mw+1):
#         u+=sp.special.comb(nr, i)
#     print(u)
#     return int(u)

# mn = 0
# s=''
# stabilizers = [None]*(lenstabs(lengthofcodeword, maxweight))
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

# print('a')

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


# step 3: generate the secret key

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


encodingkey = [None] * len(message)
for i in range(0, len(message)):
    encodingkey[i] = ''
    for q in range(0, numreps):
        if (np.random.rand() < 0.5):
            encodingkey[i] += '0'
        else:
            encodingkey[i] += '1'
with open('secretkey.txt', 'w') as file:
    for j in encodingkey:
        file.write(j + ',')
message = encode(message)
msgbin = ""
for i in range(0, len(messageog)):
    cbloc = message[i * lengthofcodeword:(i + 1) * lengthofcodeword]
    # print(int(cbloc, 2))
    # print(int(encodingkey[i],2))
    encbloc = bitxor(cbloc, encodingkey[i])
    msgbin += encbloc
# print('the encoded message with the key is '+str(msgbin))#this is the output you should send
msglen = len(msgbin)

print('b')

import random


def generate_zeroes_and_ones(n, m):
    # Generate a list with n zeroes and nm ones
    frac = int(n * m)
    elements = ['0'] * (n - frac) + ['1'] * frac

    # Shuffle the elements randomly
    random.shuffle(elements)

    # Concatenate the elements into a string
    result_string = ''.join(elements)
    return result_string


# end chatgpt code

mixer = generate_zeroes_and_ones(len(msgbin), errrate)
msgbin1 = np.binary_repr(np.bitwise_xor(int(mixer, 2), int(msgbin, 2)))
while len(msgbin1)<len(msgbin):
        msgbin1='0'+msgbin1

# with open('binarystringmeasured.txt', 'r') as g:
#    msgbin1 = g.read()
def counterr(q, v):
    count = 0
    for i in range(0, len(q)):
        if q[i] != v[i]:
            count = count + 1
    return count


print(len(msgbin))
print(len(msgbin1))
print(counterr(msgbin, msgbin1))
with open('binarystringmeasured.txt', 'w') as g:
    g.write(msgbin1)
g.close()
with open('SENDTHISONE.txt', 'w') as g:
    g.write(msgbin)
g.close()
print('runtime: ' + str(time.time() - timestart))

s1 = decode(msgbin1)
s2 = decode(msgbin)
print(counterr(s1, s2))
print(len(s1))
