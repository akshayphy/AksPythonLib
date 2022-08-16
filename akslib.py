import numpy as NP
import matplotlib.pyplot as PP
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
#*****************************************************
#
def get_spectra_fft(xx,step=1.0):
	nn = NP.size(xx)
	ps = NP.abs(NP.fft.rfft(xx))**2
	si = NP.size(ps)
	#freqs = NP.fft.rfftfreq(xx.size, step)
	freqs = NP.arange(si)
	freqs = freqs/(nn*step)
	return freqs, ps
#*****************************************************
#
def rand_unit_sphere(np,r=1.0):
	nn = NP.zeros((np,3))
	theta = (2*NP.pi)*NP.random.rand(np)
	vv = 2*NP.random.rand(np)-1
	phi = NP.arccos(vv)
	nn[:,0] = r*NP.cos(theta)*NP.sin(phi)
	nn[:,1] = r*NP.sin(theta)*NP.sin(phi)
	nn[:,2] = r*NP.cos(phi)
	return nn
#*****************************************************
#
def fit_powerlaw(xin,yin,xmin,xmax):
	ind_1 = NP.argmin(NP.abs(xin-xmin))
	ind_2 = NP.argmin(NP.abs(xin-xmax))+1
	poly = NP.polyfit(NP.log10(xin[ind_1:ind_2]),NP.log10(yin[ind_1:ind_2]),1)
	alpha = poly[0]
	a = 10**poly[1]
	yout = a*xin**alpha
	return yout,alpha,a 
#****************************************************
#
def fit_exponential(xin,yin,xmin,xmax):
  ind_1 = NP.argmin(NP.abs(xin-xmin))
  ind_2 = NP.argmin(NP.abs(xin-xmax))
  poly = NP.polyfit(xin[ind_1:ind_2],NP.log10(yin[ind_1:ind_2]),1)
  alpha = poly[0]/NP.log10(NP.exp(1.))
  a = 10**poly[1]
  yout = a*NP.exp(alpha*xin)
  return yout,alpha,a
#*****************************************************
#
def local_slope(xin,yin,xmin,xmax):
	ind_1 = NP.argmin(NP.abs(xin-xmin))
	ind_2 = NP.argmin(NP.abs(xin-xmax))
	locSl = NP.zeros(ind_2-ind_1+1)
	xout = xin[ind_1:ind_2+1]
	k = 0
	for i1 in range(ind_1,ind_2+1):
		locFit = NP.polyfit(xin[i1-2:i1+3],yin[i1-2:i1+3],1)
		locSl[k] = locFit[0]
		k = k+1
	return xout,locSl
#*****************************************************
#
def rank_order(x):
  N = NP.size(x)
  xout = NP.sort(x)
  cpdf = NP.arange(N,0,-1)/float(N)
  return xout,cpdf
#*****************************************************
#
def rank_order2(x):
  N = NP.size(x)
  xout = NP.sort(x)
  cpdf = NP.arange(0,N)/float(N)
  return xout,cpdf
#*****************************************************
#
def get_pdf_logbin(x,nbins=50,range=None):
	x = NP.log10(x)
	if range != None:
		range = NP.log10(range)
	pdf,xx = NP.histogram(x,bins=nbins,range=range)	
	xx = 10**xx
	xout = (xx[:-1]+xx[1:])/2.
	dx = NP.diff(xx)
	pdf = pdf/(dx*NP.sum(pdf)) 
	return xout,pdf
#*****************************************************
#
def get_pdf(x,nbins=50,nor=True,range=None):
	pdf,xx = NP.histogram(x,bins=nbins,density=nor,range=range)	
	xout = (xx[:-1]+xx[1:])/2. 
	return xout,pdf
#*****************************************************
#
def GKDE(xx,h=0.1,nbins=100):
	nn = NP.size(xx)
	xmin = NP.min(xx)
	xmax = NP.max(xx)
	xout = NP.linspace(xmin,xmax,nbins)
	yout = NP.zeros(nbins)
	for i1 in range(nbins):
		tmp = gaus_dist((xout[i1]-xx)/h)
		yout[i1] = NP.sum(tmp)/(nn*h)
	return xout,yout
#*****************************************************
#
def gaus_dist(xx,sigma=1.,mu=0.):
	yy = 1./(sigma*NP.sqrt(2.*NP.pi))*NP.exp(-(xx-mu)**2/(2.*sigma**2))
	return yy
#*****************************************************
#
def get_jpdf(x,y,nbins=50):
	jpdf,xx,yy = NP.histogram2d(x,y,bins=nbins,density=True)	
	xout = (xx[:-1]+xx[1:])/2. 
	yout = (yy[:-1]+yy[1:])/2. 
	return xout,yout,jpdf
#*****************************************************
#
def skwns(x):
	m1 = NP.mean(x)
	m3 = NP.mean((x-m1)**3)
	m2 = NP.mean((x-m1)**2)
	sk = m3/NP.sqrt(m2)**3
	return sk
#*****************************************************
#
def linspace_points(xin,xmin,xmax,npoints):
	xx = NP.linspace(xmin,xmax,npoints)
	indp = NP.zeros(npoints,dtype=int)
	for k in range(npoints):
		indp[k] = NP.argmin(abs(xin-xx[k]))
	return indp
#*****************************************************
#
def nice_plot(rcfontsize=20,pad=0.5,fmty=False):
	PP.rc('font',size=rcfontsize)
	if fmty:
		PP.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	PP.minorticks_on()
	PP.tick_params('both', length=8, width=2, which='major')
	PP.tick_params('both', length=4, width=1, which='minor')
	PP.tight_layout(pad=pad)
#*****************************************************
#
def get_3daxes(figno=1,fsz=(8,6)):
  fig = PP.figure(figno,figsize=fsz)
  ax = fig.add_subplot(111, projection='3d')
  return ax
#*****************************************************
#
def get_2axes2d():
	fig = PP.figure(figsize=PP.figaspect(0.5))
	ax1 = fig.add_subplot(1, 2, 1)
	ax2 = fig.add_subplot(1, 2, 2)
	PP.subplots_adjust(wspace=ws)
	return ax1,ax2
#*****************************************************
#    
def get_2axes3d():
	fig = PP.figure(figsize=PP.figaspect(0.5))
	ax1 = fig.add_subplot(1, 2, 1, projection='3d')
	PP.locator_params(nbins=3)
	ax2 = fig.add_subplot(1, 2, 2, projection='3d')
	PP.locator_params(nbins=3)
	return ax1,ax2
#*****************************************************
#
def mycmap(name='mycmap',colors=["white","blue","red"]):
    clrmp = LinearSegmentedColormap.from_list(name, colors)
    return clrmp
#*****************************************************
#
def cmap_matlab():

	cm_data = [[0.2081, 0.1663, 0.5292], 
	[0.2116238095, 0.1897809524, 0.5776761905], 
	[0.212252381, 0.2137714286, 0.6269714286], 
	[0.2081, 0.2386, 0.6770857143], 
	[0.1959047619, 0.2644571429, 0.7279], 
	[0.1707285714, 0.2919380952, 0.779247619], 
	[0.1252714286, 0.3242428571, 0.8302714286], 
	[0.0591333333, 0.3598333333, 0.8683333333], 
	[0.0116952381, 0.3875095238, 0.8819571429], 
	[0.0059571429, 0.4086142857, 0.8828428571], 
	[0.0165142857, 0.4266, 0.8786333333], 
	[0.032852381, 0.4430428571, 0.8719571429], 
	[0.0498142857, 0.4585714286, 0.8640571429], 
	[0.0629333333, 0.4736904762, 0.8554380952], 
	[0.0722666667, 0.4886666667, 0.8467], 
	[0.0779428571, 0.5039857143, 0.8383714286], 
	[0.079347619, 0.5200238095, 0.8311809524], 
	[0.0749428571, 0.5375428571, 0.8262714286], 
	[0.0640571429, 0.5569857143, 0.8239571429], 
	[0.0487714286, 0.5772238095, 0.8228285714], 
	[0.0343428571, 0.5965809524, 0.819852381], 
	[0.0265, 0.6137, 0.8135], 
	[0.0238904762, 0.6286619048, 0.8037619048], 
	[0.0230904762, 0.6417857143, 0.7912666667], 
	[0.0227714286, 0.6534857143, 0.7767571429], 
	[0.0266619048, 0.6641952381, 0.7607190476], 
	[0.0383714286, 0.6742714286, 0.743552381], 
	[0.0589714286, 0.6837571429, 0.7253857143], 
	[0.0843, 0.6928333333, 0.7061666667], 
	[0.1132952381, 0.7015, 0.6858571429], 
	[0.1452714286, 0.7097571429, 0.6646285714], 
	[0.1801333333, 0.7176571429, 0.6424333333], 
	[0.2178285714, 0.7250428571, 0.6192619048], 
	[0.2586428571, 0.7317142857, 0.5954285714], 
	[0.3021714286, 0.7376047619, 0.5711857143], 
	[0.3481666667, 0.7424333333, 0.5472666667], 
	[0.3952571429, 0.7459, 0.5244428571], 
	[0.4420095238, 0.7480809524, 0.5033142857], 
	[0.4871238095, 0.7490619048, 0.4839761905], 
	[0.5300285714, 0.7491142857, 0.4661142857], 
	[0.5708571429, 0.7485190476, 0.4493904762],
	[0.609852381, 0.7473142857, 0.4336857143], 
	[0.6473, 0.7456, 0.4188], 
	[0.6834190476, 0.7434761905, 0.4044333333], 
	[0.7184095238, 0.7411333333, 0.3904761905], 
	[0.7524857143, 0.7384, 0.3768142857], 
	[0.7858428571, 0.7355666667, 0.3632714286], 
	[0.8185047619, 0.7327333333, 0.3497904762], 
	[0.8506571429, 0.7299, 0.3360285714], 
	[0.8824333333, 0.7274333333, 0.3217], 
	[0.9139333333, 0.7257857143, 0.3062761905], 
	[0.9449571429, 0.7261142857, 0.2886428571], 
	[0.9738952381, 0.7313952381, 0.266647619], 
	[0.9937714286, 0.7454571429, 0.240347619], 
	[0.9990428571, 0.7653142857, 0.2164142857], 
	[0.9955333333, 0.7860571429, 0.196652381], 
	[0.988, 0.8066, 0.1793666667], 
	[0.9788571429, 0.8271428571, 0.1633142857], 
	[0.9697, 0.8481380952, 0.147452381], 
	[0.9625857143, 0.8705142857, 0.1309], 
	[0.9588714286, 0.8949, 0.1132428571], 
	[0.9598238095, 0.9218333333, 0.0948380952], 
	[0.9661, 0.9514428571, 0.0755333333], 
	[0.9763, 0.9831, 0.0538]]
	#
	mycmap1 = LinearSegmentedColormap.from_list('parula', cm_data)
	return mycmap1

