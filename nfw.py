import numpy as np
import scipy as sc 
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

class halo:
    rs = 1.0 #scale radius
    rhos = 1.0 #scaled density of dark matter
    rho0s = 0.1 #scaled gas density    
    Ts = 1.0 #scaled gas temperature in appropriate units

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

    def darkMatterSlice(self, x, y):
        R = np.sqrt(x*x + y*y)
        return(self.darkMatter3D(R, 0.0))

    def gasSlice(self, x, y):
        R = np.sqrt(x*x + y*y)
        return(self.gas3D(R, 0.0))

    def plotDarkMatter(self, side, sizeScale): #side is pixel number, sizeScale is (box side/rs)
        imageDarkMatter = np.zeros([side, side])
        pixels = np.linspace(-0.5*sizeScale, 0.5*sizeScale, side)
        for i in range(side):
            for j in range(side):
                x = pixels[i]
                y = pixels[j]
                R = np.sqrt(x*x + y*y)
                imageDarkMatter[i,j] = self.darkMatter3D(R, 0.0)
        return(imageDarkMatter)

    def plotGas(self, side, sizeScale, autoCorrelationScale):
        imageGas = np.zeros([side, side])
        pixels = np.linspace(-0.5*sizeScale, 0.5*sizeScale, side)
        for i in range(side):
            for j in range(side):
                x = pixels[i]
                y = pixels[j]
                R = np.sqrt(x*x + y*y)
                imageGas[i,j] = self.gas3D(R, 0.0)
        autoCorrelatedClumps = np.random.poisson(lam = np.sqrt(imageGas)) #maybe should be poisson
        autoCorrelationPixelScale = int(np.round((autoCorrelationScale/sizeScale)*side))
        kernel = np.ones([autoCorrelationPixelScale, autoCorrelationPixelScale])
        autoCorrelatedClumps = convolve(autoCorrelatedClumps, kernel)/np.sum(kernel)
        return(imageGas + 0.01*autoCorrelatedClumps)

h = halo()
plt.imshow(np.log10(h.plotDarkMatter(500, 1))) #500 side of the image in pixel, 1 side of the image in units of rs
plt.savefig("provadm.png")
plt.imshow(np.log10(h.plotGas(500, 1, 0.05))) #500 side of the image in pixel, 1 side of the image in units of rs
plt.savefig("provagas.png")
