import numpy as np
import scipy as sc 
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
np.random.seed(888)

def roundedEdgeMatrix(l):
    kernel = np.ones([l, l])
    kernel[0,0] = 0.0
    kernel[l - 1, l - 1] = 0.0
    kernel[l - 1,0] = 0.0
    kernel[0, l - 1] = 0.0
    return(kernel)

def actuallyPlotTrainingImages(dmImage, gasImage, k):
    plt.imshow(np.log10(dmImage)) 
    plt.axis('off')
    plt.savefig(str(k) + "dm.png", bbox_inches='tight')
    plt.close()
    plt.imshow(np.log10(gasImage))
    plt.axis('off')
    plt.savefig(str(k) + "gas.png", bbox_inches='tight')
    plt.close()

class halo:
    m = 1.0 #total mass
    rs = 1.0 #scale radius
    rhos = 1.0 #scaled density of dark matter
    rho0s = 0.1 #scaled gas density    
    Ts = 1.0 #scaled gas temperature in appropriate units
    xOffset = 0.0 #halo center offset from the center of the box
    yOffset = 0.0 #same in y
    autoCorrelationScale = -1

    def darkMatter3D(self, R, z):
        zs = z/self.rs
        Rs = R/self.rs
        r = np.sqrt(zs*zs + Rs*Rs)
        if r > 0.0:
        	densityFunction = self.rhos/(r*(1.0+r)*(1.0+r))
        elif r == 0.0:
        	densityFunction = np.inf
        else:
        	densityFunction = np.nan
        return(densityFunction)

    def gas3D(self, R, z):
        zs = z/self.rs
        Rs = R/self.rs
        r = np.sqrt(zs*zs + Rs*Rs)
        b = self.rhos*self.rs*self.rs/self.Ts
        if r > 0.0:
            densityFunction = self.rho0s*np.exp(-b)*np.power((1.0 + r),(b/r))
        elif r == 0.0:
        	densityFunction = self.rho0s
        else:
        	densityFunction = np.nan
        return(densityFunction)

    def plotDarkMatter(self, side, sizeScale): #side is pixel number, sizeScale is (box side/rs)
        ox = self.xOffset
        oy = self.yOffset
        imageDarkMatter = np.zeros([side, side])
        pixels = np.linspace(-0.5*sizeScale, 0.5*sizeScale, side)
        for i in range(side):
            for j in range(side):
                x = pixels[i]
                y = pixels[j]
                R = np.sqrt((x-ox)*(x-ox) + (y-oy)*(y-oy))
                imageDarkMatter[i,j] = self.darkMatter3D(R, 0.0)
        return(self.m*imageDarkMatter)

    def plotGas(self, side, sizeScale):
        ox = self.xOffset
        oy = self.yOffset
        autoCorrelationScale = self.autoCorrelationScale
        imageGas = np.zeros([side, side])
        pixels = np.linspace(-0.5*sizeScale, 0.5*sizeScale, side)
        for i in range(side):
            for j in range(side):
                x = pixels[i]
                y = pixels[j]
                R = np.sqrt((x-ox)*(x-ox) + (y-oy)*(y-oy))
                imageGas[i,j] = self.gas3D(R, 0.0)
        if(autoCorrelationScale > 0):
            autoCorrelatedClumps = np.random.poisson(lam = np.sqrt(imageGas)) 
            autoCorrelationPixelScale = 1 + int((autoCorrelationScale/sizeScale)*side)
            kernel = roundedEdgeMatrix(autoCorrelationPixelScale)
            autoCorrelatedClumps = convolve(autoCorrelatedClumps, kernel)/np.sum(kernel)
            imageGas = imageGas + 0.05*autoCorrelatedClumps/self.Ts
        return(self.m*imageGas)

    def randomize(self, boxSize, minimumrs):
        r = np.random.uniform(size = 6)
        self.m = r[0]
        self.xOffset = -boxSize//2 + boxSize*r[1]
        self.yOffset = -boxSize//2 + boxSize*r[2]
        self.rs = minimumrs+r[3]
        self.Ts = minimumrs+r[4]
        self.autoCorrelationScale=minimumrs/2.0+r[5]

    def printHalo(self, j):
        print(" halo " + str(j) + " has m=" + str(self.m) + " rs=" + str(self.rs) + " Ts = " + str(self.Ts))


Nimages = 1000
maxHalosPerImage = 5
side = 256
boxSize = 5 #image side in units of the halo rs
minimumrs = 0.5 #minimum value of rs (choose something > 0 and smaller than the boxSize)

h = halo()
for k in range(Nimages):
    gasImage = np.zeros([side, side])
    dmImage = np.zeros([side, side])
    NhalosPerImage = np.random.randint(1, maxHalosPerImage+1)
    print("Image " + str(k) + " contains " + str(NhalosPerImage) + " haloes")
    for j in range(NhalosPerImage):
        h.randomize(boxSize, minimumrs)
        h.printHalo(j)
        dmImage += h.plotDarkMatter(side, boxSize)
        gasImage += h.plotGas(side, boxSize)
    actuallyPlotTrainingImages(dmImage, gasImage, k)
