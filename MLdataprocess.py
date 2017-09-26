import numpy as np
import pandas as pd


#get amino acid number of VPU
def AAcheck(sequence, name):
    AAdict = pd.read_csv("aminoacidsdict.csv")
#    print AAdict
    for n,letter in enumerate(sequence):
        for i in range(0, len(AAdict)):
            if letter == AAdict[name][i]:
                sequence[n] = AAdict['numref'][i]
    return sequence

def openHbond(filename):
    with open(filename) as f:
        lst = []
        for line in f:
            lst += [line.split()]
    #select a specifit column
    #column1 = [x[0] for x in lst]

    #change *.dat file to dataframe
    Htable = pd.DataFrame(lst)

    #change the header of the dataframe
    new_header = Htable.iloc[0]
    Htable = Htable[1:]
    Htable.columns = new_header
    Htable.index = range(len(Htable))
    return Htable

#function to read acceptor and donor information
def getHbond(filname, type):
    table = openHbond(filname)
    if type == "Acceptor":
        name = "#Acceptor"
    else:
        name = "Donor"
    letter = []
    pos = []
    for i in range(len(table)):
        letterL = table[name][i][0:3]
        posP = table[name][i][4:7]
        letter.append(letterL)
        pos.append((int(posP) - 373))
    return [letter, pos]



