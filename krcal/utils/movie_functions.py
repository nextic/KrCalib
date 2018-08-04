"""

"""


import imageio
import matplotlib.image as mpimg

def movie(label, nbins, duration = 0.5):
    imgs = [imageio.imread(label+str(i)+'.png') for i in range(nbins)]
    imageio.mimsave(label+'.gif', imgs, duration = duration)
