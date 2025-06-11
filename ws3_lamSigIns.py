#=================================================================
# AE2220-II - Computational Modelling.
# Analysis program for Worksession 3 - Insulate
#
# Line 20: Definition of gammas for 4-stage time march
# Line 31: Definition of the lambda-sigma relation
#
#=================================================================
import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#------------------------------------------------------
# Input parameters
#------------------------------------------------------
nx    = 20;    # Number of mesh points (must be even)
g     = 0.4;   # nu*(Delta t)/(Delta x)^2
nu    = 1.0;   # Thermal diffusivity
a     = 0.7;   # LSRK : coefficient of first stage
b     = 0.8;   # LSRK : coefficient of second stage
c     = 0.9;   # LSRK : coefficient of third stage
d     = 1.0;   # LSRK : coefficient of fourth stage


#------------------------------------------------------
# Function for the lambda-sigma relation
#------------------------------------------------------
def lamSig(ldt):
  L = ldt
  sigma = 1 + ldt;               # Euler explicit time march
  #sigma = 1/(1- ldt);            # Euler implicit time march
  sigma =  1 + L**4*a*b*c*d+L**3*b*c*d+L**2*c*d+L*d              # ADD THE RKLS LAMBDA-SIGMA RELATION HERE**
  return sigma

#------------------------------------------------------
# Define the semi-discrete matrix A * Dt 
# for linear diffusion * (Delta x)^2 
#------------------------------------------------------
ADt = np.zeros((nx, nx))
for i in range(0, nx):
   if i == 0:      # Left periodic boundary
      ADt[i,nx-1] =  1;
      ADt[i,i]    = -2;
      ADt[i,i+1]  =  1;
   elif i == nx-1: # Right periodic boundary
      ADt[i,i-1]  =  1;
      ADt[i,i]    = -2;
      ADt[i,0]    =  1;
   else :          # Interior
      ADt[i,i-1]  =  1;
      ADt[i,i]    = -2;
      ADt[i,i+1]  =  1;
ADt *= g;

#------------------------------------------------------------
# Compute the semi-discrete eigenvalues lambda *DT
# from the expression in the notes for circulant matrices
#------------------------------------------------------------
beta=np.zeros(nx);
ldt=np.zeros(nx,'complex')
for m in range(nx):
  beta[m] = 2*np.pi*m/nx;
  if beta[m] > np.pi: beta[m] = 2*np.pi - beta[m]; # negative beta modes
  for j in range(0, nx):
    ldt[m] = ldt[m] + ADt[0,j]*np.exp(1j*2.*np.pi*j*m/nx);

#------------------------------------------------------------
# Compute the eigenvalues of C using the lambda-sigma relation, 
# then determine the amplitude and phase errors relative 
# to their exact values for linear diffusion
#------------------------------------------------------------
sigma  = lamSig(ldt);
ampErr = np.zeros(nx);
pseErr = np.zeros(nx);
for m in range(1,nx):
   ampErr[m] = np.exp(-g*beta[m]*beta[m]) - np.abs(sigma[m]) ; 
   pseErr[m] = np.angle(sigma[m]); 
   if (np.abs(pseErr[m])>3.14): pseErr[m] = 0.
   if (m>nx/2): pseErr[m] = -pseErr[m] # negative beta modes
   


#===================================================================
# Output results
#===================================================================

#------------------------------------------------------------
# Write the results to the screen
#------------------------------------------------------------
print('mode   beta       ldt          sigma       ampErr  pseErr')
print("----------------------------------------------------------")
for m in range(nx):
  print( '%3i' % m, '| %5.2f' % beta[m], '| %5.2f' % np.real(ldt[m]), '%5.2f' % np.imag(ldt[m]), \
          '| %5.2f' % np.real(sigma[m]), '%5.2f' % np.imag(sigma[m]), '| %7.4f' % ampErr[m], '%7.4f' % pseErr[m] )        

#------------------------------------------------------
# Define a grid of points on the lambda*Dt plane
# then compute |sigma| at these points 
# so we can plot contours of |sigma| < 1 on the
# lambda*Dt plane.
#------------------------------------------------------
prr = np.linspace(-3.0,0.2,50)   # Real range
pir = np.linspace(-3.0,3.0,50)   # Imagainary range
prc,pic = np.meshgrid(prr,pir);  # Grid of points
pldt=prc + 1j*pic;               # lambda dt values for each point
pSigma = lamSig(pldt);           # sigma at each point
pMagSig = np.abs(pSigma);        # |sigma at each point|

#------------------------------------------------------
# Plot values on the ldt plane and performance vs beta
#------------------------------------------------------
fig = plt.figure(figsize=(15,10))

# Plot the lambda-delta t plane
ax1 = fig.add_subplot(121)
ax1.set_title("$|\\sigma|$ contours and $\\lambda_m$ of [A] on the $\\lambda\\Delta t$ plane")
ax1.set_xlabel("$Re(\\lambda\\Delta t)$")
ax1.set_ylabel("$Im(\\lambda\\Delta t)$")
ax1.grid()
lev = np.linspace(0.0,1.0,11)
a = ax1.contour(prc, pic, pMagSig, lev)
fig.colorbar(a, ax=ax1)
ax1.plot(np.real(ldt),np.imag(ldt),'ro',markersize=8,label="$\\Delta t *(\\lambda_m \\; $of$ \\; [A])$")
ax1.legend(loc="lower left")

# Plot the amplitude and phase error
ax2 = fig.add_subplot(122)
ax2.set_title("Performance vs wave number")
ax2.set_xlabel("$\\beta$")
ax2.set_xlim((0,np.pi))
ax2.grid()
ax2.plot(beta, ampErr, '-bo', label="Amplitude error")
ax2.plot(beta, pseErr, '-ro', label="Phase error")
ax2.legend(loc="lower left")

plt.show()

