import numpy as NP
import matplotlib.pyplot as PP
PP.ion()
# import akslib as aks

PP.rc('figure',figsize=(3.5,2.625),dpi=300)
PP.rc('savefig',bbox='tight',pad_inches=0.05,dpi=600)
PP.rc('axes',linewidth=0.7,grid=False)
PP.rc('lines',linewidth=1.,markersize=3.)
PP.rc('grid',linewidth=0.5)
PP.rc('xtick',direction='in',top=True)
PP.rc('ytick',direction='in',right=True)
PP.rc('xtick.major',size=4,width=0.7)
PP.rc('xtick.minor',visible=True,size=2,width=0.4)
PP.rc('ytick.major',size=4,width=0.7)
PP.rc('ytick.minor',visible=True,size=2,width=0.4)
PP.rc('font',size=9)
PP.rc('legend',fontsize=7,frameon=True)
PP.rc('text',usetex=True)
