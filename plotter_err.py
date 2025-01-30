fnames = ("75 MHz Key Steg Homodyne", "75 MHz No Key Steg Homodyne")
import numpy as np
import matplotlib.pyplot as plt
colors = ("k", "r")
for i in range(len(fnames)):
    with open(fnames[i], 'r') as f:
        ppp = f.read()

    A = (ppp.split(","))

    Adoub = np.ones(len(A) - 1)
    idx = 0
    for a in A:
        try:
            Adoub[idx] = np.double(a)
            idx += 1
        except(ValueError):
            print("warning probs j end of file and nbd")

    nr = np.arange(1, 55, 2)
    print(nr[25:26])
    plt.plot(nr, Adoub, colors[i])
plt.legend(("Vertical Angle", "Distribution"))
plt.grid()
plt.xlabel("Number of Repetitions (n)")
plt.ylabel("Error Rate")
plt.ylim(1E-3, 0.5)
plt.xlim(0, 40)
plt.yscale("log")
plt.savefig("exponential.png")
