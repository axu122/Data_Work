import numpy as np


#p=3 --> y= ax^3 + bx^2 + cx + d
def generatePoly(degTarget):
    coefs = []
    for _ in range(degTarget):
        coefs.append(np.random.normal(scale = 1.0))
    coef_sum = 0
    print(coefs)
    for coef in coefs:
        coef_sum += coef**2
    coefs = np.true_divide(coefs,np.sqrt(coef_sum))
        
    return coefs




def generateRandSamp(numPoints, degTarget, noise):
    X = []
    y = []
    for _ in range(numPoints):
        X.append(np.random.uniform(-1,1))
        y.append(generatePoly(degTarget) + np.random.normal(scale = noise))
    return X,y
    
def oneFit(numPoints=10, degTarget=4, degFit=2, noiseStd=0.1):
    generatePoly(degTarget)
    X,y = generateRandSamp(numPoints,degTarget, noiseStd)
    p = np.polyfit(X, y, deg = degTarget)
    print(p)
    
oneFit()