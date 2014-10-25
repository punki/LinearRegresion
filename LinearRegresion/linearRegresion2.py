import time, random, numpy as np


def booleanClassified(x, y, f):
    ySep = f(x)
    if y >= ySep:
        return 1
    else:
        return -1


# .|0 |1
# 0|x1|y1
# 1|x2|y2
# return funkcja
def f(fCoordinate):
    a = (fCoordinate[0, 1] - fCoordinate[1, 1]) / (fCoordinate[0, 0] - fCoordinate[1, 0])
    b = fCoordinate[1, 1] - a * fCoordinate[1, 0]
    return lambda x: a * x + b


def fNonLinear():
    return lambda x1, x2: np.sign(x1 * x1 + x2 * x2 - 0.6)


def applayFunction(samples, function):
    result = []
    for (x1, x2) in samples[:, 1:3]:
        result.append(function(x1, x2))
    return result


# samples (x0,x1,x2)
def classifiedAll(samples, f):
    classes = []
    for (x, y) in samples[:, 1:3]:
        classes.append(booleanClassified(x, y, f))
    return classes


def countDiffElements(h, targetClasses):
    diffElements = 0
    for idx, he in enumerate(h):
        if targetClasses[idx] != he:
            diffElements += 1
    return diffElements


def compMiss(hn, fn):
    miss = []
    for idx, hne in enumerate(hn):
        if fn[idx] != hne:
            miss.append(idx)
    return miss


def transformSamples(samples, rows):
    result = np.ones((rows, 6), float)
    for idx, s in enumerate(samples):
        result[idx] = [s[0], s[1], s[2], s[1] * s[2], s[1] * s[1], s[2] * s[2]]
    return result

def noise(data, elements):
    noiseIdx = set()
    while len(noiseIdx) < elements * 0.1:
        noiseIdx.add(random.randint(0, elements - 1))
    for i in noiseIdx:
        data[i] *= -1

def perceptron(g, samples, targetClasses):
    iter = 1
    while True:
        hPerc = map(lambda e: 1 if e >= 0 else -1, samples.dot(g))
        miss = compMiss(hPerc, targetClasses)
        if miss == []:
            break
        mIdx = random.choice(miss)
        # learn
        g[1] = g[1] + targetClasses[mIdx] * samples[mIdx, 1]
        g[2] = g[2] + targetClasses[mIdx] * samples[mIdx, 2]
        g[0] += targetClasses[mIdx] * 1
        iter += 1
    return iter

# ############# MAIN ##############
tt = time.time()
n, N, nOut, nNonLinear = 10, 1000, 1000, 1000
gs = []
gsNonLinear = []
gsNonLinearTrans = []
Ein = 0
EinNL = 0
EinNLTrans = 0
Eout = 0
EoutNL = 0
percIter = 0
for step in range(N):
    # kolumny 0=x0 1=x1 2=x2
    fCoordinate = np.random.uniform(-1.0, 1.0, (2, 2))
    samples = np.ones((n, 3), float)
    samples[:, 1:3] = np.random.uniform(-1.0, 1.0, (n, 2))
    fFun = f(fCoordinate)
    targetClasses = classifiedAll(samples, fFun)
    g = np.linalg.pinv(samples).dot(targetClasses)
    gs.append(g)
    hReal = samples.dot(g)
    h = map(lambda e: 1 if e >= 0 else -1, hReal)
    Ein += countDiffElements(h, targetClasses) / float(n)
    # print 'In step={0} \n w={1} \n targetClasses={2} \n h={3} \n hReal={5} \n Ein={4}'.format(step, g, targetClasses, h,
    # Ein, hReal)

    # cwiczenie 6
    samplesOut = np.ones((nOut, 3), float)
    samplesOut[:, 1:3] = np.random.uniform(-1.0, 1.0, (nOut, 2))
    targetClassesOut = classifiedAll(samplesOut, fFun)
    hRealOut = samplesOut.dot(g)
    hOut = map(lambda e: 1 if e >= 0 else -1, hRealOut)
    Eout += countDiffElements(hOut, targetClassesOut) / float(nOut)

    # cwiczenie 7
    percIter += perceptron(np.copy(g), samples, targetClasses)

    # cwiczenie 8 Non Linear Transform
    samplesNL = np.ones((nNonLinear, 3), float)
    samplesNL[:, 1:3] = np.random.uniform(-1.0, 1.0, (nNonLinear, 2))
    targetClassesNL = applayFunction(samplesNL, fNonLinear())
    # noise
    noise(targetClassesNL, nNonLinear)

    gNL = np.linalg.pinv(samplesNL).dot(targetClassesNL)
    gsNonLinear.append(gNL)
    hRealNL = samplesNL.dot(gNL)
    hNL = map(lambda e: 1 if e >= 0 else -1, hRealNL)
    EinNL += countDiffElements(hNL, targetClassesNL) / float(nNonLinear)
    # print 'targetClassesNL={} noiseIdx={} EinNL={}'.format(targetClassesNL, noiseIdx, EinNL)

    # cwiczenie 9
    samplesNLTrans = transformSamples(samplesNL, nNonLinear)
    gNLTrans = np.linalg.pinv(samplesNLTrans).dot(targetClassesNL)
    gsNonLinearTrans.append(gNLTrans)
    hRealNLTrans = samplesNLTrans.dot(gNLTrans)
    hRealNLTrans = map(lambda e: 1 if e >= 0 else -1, hRealNLTrans)
    EinNLTrans += countDiffElements(hRealNLTrans, targetClassesNL) / float(nNonLinear)
    # print 'samplesNLTrans={}'.format(samplesNLTrans)

    #cwiczenie 10
    samplesNLOut = np.ones((nOut, 3), float)
    samplesNLOut[:, 1:3] = np.random.uniform(-1.0, 1.0, (nOut, 2))
    targetClassesNLOut = applayFunction(samplesNLOut, fNonLinear())
    noise(targetClassesNLOut, nOut)
    samplesNLTransOut = transformSamples(samplesNLOut, nOut)
    hRealOutNL = samplesNLTransOut.dot(gNLTrans)
    hOutNL = map(lambda e: 1 if e >= 0 else -1, hRealOutNL)
    EoutNL += countDiffElements(hOutNL, targetClassesNLOut) / float(nOut)

EinAvg = Ein / float(N)
EOutAvg = Eout / float(N)
EOutAvgNL = EoutNL / float(N)
percIterAvg = percIter / float(N)
EinNLAvg = EinNL / float(N)
EinNLTransAvg = EinNLTrans / float(N)
print 'Avg Ein={0} Ein={1} || EOutAvg={3} Eout={2} || percIterAvg={4} || EinNLAvg={5} || EinNLTransAvg={6} EOutAvgNL={7}'.format(
    EinAvg, Ein, Eout, EOutAvg, percIterAvg, EinNLAvg, EinNLTransAvg,EOutAvgNL)
# print '\n\ngsNonLinearTrans={}'.format(gsNonLinearTrans)
# cwiczenie 6
