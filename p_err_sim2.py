import numpy as np
from PIL import Image
import lzma
from lzma import FORMAT_XZ
import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt


def generate_zeroes_and_ones(n, m):
    # Generate a list with n zeroes and nm ones
    frac = int(n * m)
    elements = ['0'] * (n - frac) + ['1'] * frac

    # Shuffle the elements randomly
    random.shuffle(elements)

    # Concatenate the elements into a string
    result_string = ''.join(elements)
    return result_string


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


def encode(msg, numreps):
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


def compute_error_probability_repetition(prob, nn):
    # sum over all weights of errors with appropriate probabilities
    zed = 0
    for i in range(int(nn / 2) + 1, nn + 1):
        zed += sp.special.comb(nn, int(nn / 2) + 1) * prob ** (int(nn / 2) + 1) * (1 - prob) ** (int(nn / 2))
    return zed


nr = np.arange(1, 55, 2)
bl = np.ones(np.size(nr))
idxxx2 = -1
fnames = ("10 MHz BPSK", "75 MHz Key Steg Homodyne", "75 MHz No Key Steg Homodyne")
for errrate in (.23, .21, .27):
    idxxx = -1
    idxxx2 += 1
    for blocksize in nr:
        idxxx += 1
        usinglzma = False
        # add errors if desired

        a, w, h = png_to_binary('resized.png')
        with open('widthheight.txt', 'w') as f:
            f.write(str(w) + ',' + str(h))
        f.close()

        z = a
        if (usinglzma):
            loh = 5  # length of header
            b = lzma.compress(a.encode(), format=FORMAT_XZ, preset=1)
            bl = len(b) - loh  # 1479
            z = bin(int.from_bytes(b[loh:], byteorder='big'))

            z = z[2:]  # eliminate b'

        # step 1: choose a message and a channel
        messageog = z
        message = messageog

        order = 0
        # blocksize = 2 ** m
        print(blocksize)
        numreps = blocksize
        lengthofcodeword = blocksize

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
        file.close()
        message = encode(message, numreps)
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

        mixer = generate_zeroes_and_ones(len(msgbin), errrate)
        msgbin1 = np.binary_repr(np.bitwise_xor(int(mixer, 2), int(msgbin, 2)))
        while len(msgbin1) < len(msgbin):
            msgbin1 = '0' + msgbin1


        def counterr(q, v):
            count = 0
            for i in range(0, len(q)):
                if q[i] != v[i]:
                    count = count + 1
            return count


        with open('binarystringmeasured.txt', 'w') as g:
            g.write(msgbin1)
        g.close()
        with open('binarystringmeasured.txt', 'r') as f:
            ppp = f.read()
        f.close()
        measuredbits = ppp  # put whatever you measured here. should be a binary message

        print(counterr(msgbin1, msgbin) / len(msgbin))
        with open('secretkey.txt', 'r') as f:
            rawstring = f.read()
        f.close()
        encodingkey = rawstring.split(',')[0:-1]

        # step 8: reverse the key
        messagedecoded = ""
        for i in range(0, len(z)):
            encbloc = bitxor(measuredbits[i * lengthofcodeword:(i + 1) * lengthofcodeword], encodingkey[i])
            # print(len(encbloc))
            messagedecoded += encbloc
        # step 9: decode the message
        messagedecoded = decode(messagedecoded)

        print(counterr(messagedecoded, z) / len(messagedecoded))
        bl[idxxx] = counterr(messagedecoded, z) / len(messagedecoded)

    with open(fnames[idxxx2], "w") as txt_file:
        for line in bl:
            txt_file.write(str(line)+ ",")
    plt.plot(nr, bl)

plt.legend(fnames)
plt.grid()
plt.xlabel("Number of Repetitions (n)")
plt.ylabel("Error Rate")
plt.savefig("test.png")

plt.yscale("log")
plt.savefig("exponential.png")
