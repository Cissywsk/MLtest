import numpy as np
import pandas as pd
import MLdataprocess as mld

#get sequence of VPU
VPU = "AIVALVVAIIIAIVVWSIV"
sepVPU = list(VPU)


def getHmap(Hbondfile):
	#call the function
	VPU = "AIVALVVAIIIAIVVWSIV"
	sepVPU = list(VPU)
	acc = mld.getHbond(Hbondfile, "Acceptor")
	don = mld.getHbond(Hbondfile, "Donor")
	HbdTable = mld.openHbond(Hbondfile)

	numVPU = mld.AAcheck(sepVPU, 'singleC')
	accnum = mld.AAcheck(acc[0], 'threeC')
	donnum = mld.AAcheck(don[0], 'threeC')

	#generate heatmap
	Hmap = np.zeros((19, 19))

	n = 0
	for i,j in zip(acc[1], don[1]):
	    Hmap[i][j] = HbdTable['Frac'][n]
	    n += 1

	#attach VPU structure info in both axis
	HmapVpu = np.vstack((numVPU, Hmap))
	numVPU.insert(0, -1)
	numVPU = np.array(numVPU)
	numVPU = numVPU.reshape((len(numVPU), 1))
	HmapVPU = np.hstack((numVPU, HmapVpu))
	oneDHmap = HmapVPU.reshape((1, 400))
	oneDHmap = oneDHmap[0]
	return oneDHmap

#start grabbing X input data
startfile = 6744
endfile = 6744 + 32080
Xinput = []
for k in range(startfile, endfile, 80):
	Hbondfile = "kink_cluster"+str(k)+".dat"
	print("get information of file "+Hbondfile)
	Xresult = getHmap(Hbondfile)
	Xinput.append(Xresult)

kinkinfo = pd.read_csv("kinkinfo.csv")
Yinput = kinkinfo['Kink_active']

print("starting Machine Learning analysis")
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
for i in range(1, 10):
	knn = KNeighborsClassifier(n_neighbors=i)
	knn.fit(Xinput[0:320], Yinput[0:320])
	Y_pred = knn.predict(Xinput[320:])
	knn_score = metrics.accuracy_score(Yinput[320:], Y_pred)
	print("knn_score:"+str(knn_score))

from sklearn import svm
from sklearn.model_selection import cross_val_score

for cost in [1, 3, 5, 10, 50, 100, 300, 500, 1000]:
	clf = svm.SVC(kernel='linear', C=cost)
	svm_scores = cross_val_score(clf, Xinput, Yinput, cv=5)
	print("svm cross validation score with cost "+str(cost))
	print(svm_scores)
