from PIL import Image
from encodetrojan import png_to_binary, lenstabs
from encoding import encode
import random
import matplotlib.pyplot as plt
import lzma
from lzma import FORMAT_XZ
import numpy as np
import scipy as sp


def counterr(q, v):
    count = 0
    for i in range(0, len(q)):
        if q[i] != v[i]:
            count = count + 1
    return count


def decodeCodeword(m):
    if (m.count('0') > m.count('1')):
        return '0'
    return '1'


def decode(msg):
    s = ""
    for i in range(0, int(len(msg) / lengthofcodeword)):
        s += decodeCodeword(msg[lengthofcodeword * i:lengthofcodeword * (i + 1)])
    return s


errcount = 0
idxx = -1
nrl =range(1, 20)
perrpostdecode = np.ones(len(nrl))
for numreps in nrl:
    idxx += 1
    print("NUMREPS:")
    print(numreps)
    lengthofcodeword = numreps

    image = Image.open('bw_trojan.png')
    new_image = image.resize((150, 150))
    new_image.save('resized.png')

    a, w, h = png_to_binary('resized.png')
    with open('widthheight.txt', 'w') as f:
        f.write(str(w) + ',' + str(h))
    f.close()

    message = a
    messageog = a
    maxweight = int(numreps / 2)

    mn = 0
    s = ''
    stabilizers = [0] * (lenstabs(numreps, maxweight))
    for i in range(0, maxweight):
        for k in range(0, numreps):
            if i == 0:
                for j in range(0, k):
                    s = '0' * j
                    s += '1'
                    s += '0' * (k - j - 1)
                    s += '1'
                    s += '0' * (numreps - k - 1)
                    stabilizers[mn] = s
                    mn += 1
            else:
                s = '0' * k
                s += '1'
                s += '0' * (numreps - k - 1)
                stabilizers[mn] = s
                mn = mn + 1

    stabilizers[-1] = '0' * numreps

    stbs = ['0'] * (2 * len(stabilizers))
    for i in range(0, len(stabilizers)):
        stbs[i] = stabilizers[i]
        stbs[len(stabilizers) + i] = np.binary_repr(
            np.bitwise_xor(int('1' * lengthofcodeword, 2), int(str(stabilizers[i]), 2)), width=lengthofcodeword)
    stabilizers = stbs
    nstabs = len(stabilizers)
    encodingkey = [0] * len(message)
    for i in range(0, len(message)):
        stab = np.random.randint(0, nstabs)
        encodingkey[i] = stabilizers[stab]

    with open('secretkey.txt', 'w') as file:
        for j in encodingkey:
            file.write(str(j) + ',')
    message = encode(message)

    msgbin = ""
    for i in range(0, len(messageog)):
        try:
            cbloc = message[i * lengthofcodeword:(i + 1) * lengthofcodeword]
            encbloc = np.binary_repr(np.bitwise_xor(int(cbloc, 2), int(encodingkey[i], 2)), width=lengthofcodeword)
            msgbin += encbloc
        except(TypeError, ValueError):
            errcount += 1
    msgbinerr = list(msgbin)
    p_err = .1
    n_err = len(msgbin) * p_err
    deck = list(range(1, len(msgbin)))
    random.shuffle(deck)

    for i in range(0, int(n_err)):
        h = deck.pop()
        # print(a)
        if msgbin[h] == "1":
            msgbinerr[h] = "0"
        else:
            msgbinerr[h] = "1"
    msgbinerr = ''.join(msgbinerr)
    countright = 0
    for i in range(0, len(msgbin)):
        if msgbin[i] == msgbinerr[i]:
            countright += 1

    print("PCD")
    print(countright / len(msgbin))
    print("PERR")
    print(1 - countright / len(msgbin))

    msglen = len(msgbin)
    print(len(msgbinerr))
    with open('simulated.txt', 'w') as g:
        g.write(msgbinerr)

    with open('widthheight.txt') as f:
        listofthings = f.read().split(',')
        w = listofthings[0]
        h = listofthings[1]
    f.close()

    with open('simulated.txt', 'r') as f:
        ppp = f.read()

    measuredbits = ppp
    nstabs = len(stabilizers)
    with open('secretkey.txt', 'r') as f:
        rawstring = f.read()
    encodingkey = rawstring.split(',')[0:-1]
    messagedecoded = ""
    z = a

    # step 8: reverse the key
    for i in range(0, len(z)):
        # print(int(measuredbits[i*lengthofcodeword:(i+1)*lengthofcodeword], 2))
        # print(i)
        try:
            encbloc = np.bitwise_xor(int(measuredbits[i * lengthofcodeword:(i + 1) * lengthofcodeword], 2),
                                     int(encodingkey[i], 2))
            messagedecoded += np.binary_repr(encbloc, lengthofcodeword)
        except ValueError:
            errcount += 1
    # step 9: decode the message
    messagedecoded = decode(messagedecoded)
    print("Post coding PERR")
    print(counterr(messagedecoded, z) / len(messagedecoded))
    perrpostdecode[idxx] = counterr(messagedecoded, z) / len(messagedecoded)
plt.plot(nrl, perrpostdecode)
plt.show()
