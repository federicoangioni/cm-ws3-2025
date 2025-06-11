#=================================================================
# AE2220-II Computational Modelling
# In-class code for worksession 3 - Insulate
#=================================================================

import math as math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as pltani
from scipy.optimize import minimize


#--------------------------------------------------------------------
# Function to set material properties (Do not edit)
#--------------------------------------------------------------------
mat_rho   = np.zeros(3)
mat_nu    = np.zeros(3)
mat_name  = ['' for i in range(3)]
mat_color = ['' for i in range(3)]
plots = 0

def setMaterial(number, name):

 if (number>2) : print('Material number must be <=2'); exit();
 
 if (name=='Glass'):
   mat_rho  [number] = 2210;          # ULE Glass - density
   mat_nu   [number] = 7.9*10**(-7);  # ULE Glass - diffusivity
   mat_name [number] = name;          
   mat_color[number] = '#1f77b4'
 elif (name=='Aluminium'):
   mat_rho  [number] = 2643;          # Aluminium - density
   mat_nu   [number] = 9.7*10**(-5);  # Aluminium - diffusivity
   mat_name [number] = name;          
   mat_color[number] = 'orangered'
 elif (name=='Al6061T6'):
   mat_rho  [number] = 2700;          # Al6061T6 - density
   mat_nu   [number] = 4.4*10**(-5);  # Al6061T6 - diffusivity
   mat_name [number] = name;          
   mat_color[number] = 'peru'
 elif (name=='Steel'):
   mat_rho  [number] = 8000;          # Steel, stainless 304A - density
   mat_nu   [number] = 4.2*10**(-6);  # Steel, stainless 304A - diffusivity
   mat_name [number] = name;
   mat_color[number] = 'lime'
 elif (name=='Titanium'):
   mat_rho  [number] = 4506;          # Titanium - density
   mat_nu   [number] = 8.85*10**(-6); # Titanium - diffusivity
   mat_name [number] = name;          
   mat_color[number] = 'gold'
 else:
   print('Material name',name,' unknown'); exit();
   
 return

setMaterial(0,'Glass')



#------------------------------------------------------
#  Design section (Can edit this section)
#------------------------------------------------------
setMaterial(1,'Steel')
setMaterial(2,'Aluminium')

r1         = 21e-3;              # Radius of glass (fixed)
r3         = 195e-3;             # Outer radius of material 3 (fixed)

def optimize_mass(vars):

  # r2         = 100e-3;     # Outer radius of green material (r1 < r2 <r3)
  r2= vars;


  #------------------------------------------------------
  # Mass per unit length 
  #------------------------------------------------------
  mass = np.pi*r1*r1*mat_rho[0] \
      + np.pi*(r2*r2-r1*r1)*mat_rho[1] \
      + np.pi*(r3*r3-r2*r2)*mat_rho[2]
  
  return mass

#------------------------------------------------------
# Prepare for time march
#------------------------------------------------------



def temperature_constraint(vars):
  r2 = vars
  
  t          = 0;                               # Initial time
  u_clAmp    = 0;                               # Initial centerline temperature amplitude
  #setMaterial(2,'Titanium')
  #setMaterial(2,'Aluminium')
  #setMaterial(2,'Al6061T6')


  #------------------------------------------------------
  # Discretisation parameters  (Can edit this section)
  #------------------------------------------------------
  nr         = 25                            # Number of mesh points in r
  gMax       = 0.3;                          # Max nu*Delta t/ Delta r^2
  gmlsrk     = np.array([0.1, 0.2, 0.5, 1])  # gm1, gm2, gm3, gm4 for LSRK


  #------------------------------------------------------
  #  Plotting options (Can edit this section)
  #------------------------------------------------------
  plot_skip  = 20;        # Time step skip for static plots
  scr_skip   = 500;       # Time step skip for screen output
  ani_count  = 10;        # Number of images per period in animation


  #------------------------------------------------------
  # Simulation parameters (Do not change)
  #------------------------------------------------------
  r1         = 21e-3;              # Radius of glass (fixed)
  r3         = 195e-3;             # Outer radius of material 3 (fixed)
  rmt        = r3/50.;             # Delta r for material transitions.
  f_amp      = 3.3e-4;             # External forcing amplitude
  f_freq     = 1e-4;               # External forcing frequency
  cli        = 1;                  # Centreline grid index
  u_amb      = 284;                # Ambient temperature
  t_amp      = 5e4;                # time after which amplitude is measured
  t_end      = t_amp+2/f_freq;     # Simulation end time (2 periods after t_amp)


  #------------------------------------------------------
  # Make mesh and initialise solution and plotting vectors 
  #------------------------------------------------------
  r      = np.linspace(0, r3,  nr)     # Mesh coordinate vector
  nu     = np.linspace(0, 1.0, nr)     # Diffusivity vector
  g      = np.linspace(0, 1.0, nr)     # Local nu*deltaT/deltaX^2
  u_n    = np.ones(nr)*u_amb;          # Temperature at time level n
  u_st   = np.ones(nr)*u_amb;          # Temperature for RK stage
  u_np1  = np.ones(nr)*u_amb;          # Temperature at time level n+1              
  

  #------------------------------------------------------
  # Set local diffusivity values
  #------------------------------------------------------
  delNu1=(mat_nu[1]-mat_nu[0]);
  delNu2=(mat_nu[2]-mat_nu[1]);
  for i in range(nr):
    if (r[i] < r1-rmt):
      nu[i]  = mat_nu[0];
    elif (r[i] < r1+rmt):
      mtang=(r[i]-(r1-rmt))*math.pi/(2.*rmt);
      nu[i]  = mat_nu[0] + (0.5-math.cos(mtang)/2.)*delNu1;
    elif (r[i] < r2-rmt):
      nu[i]  = mat_nu[1];
    elif (r[i] < r2+rmt):
      mtang=(r[i]-(r2-rmt))*math.pi/(2.*rmt);
      nu[i]  = mat_nu[1] + (0.5-math.cos(mtang)/2.)*delNu2;
    else:
      nu[i]  = mat_nu[2];


  #------------------------------------------------------
  # Compute time march parameters
  #------------------------------------------------------
  maxnu    = max(nu);                     # Max diffusivity encountered               
  deltaR   = r3/(nr-1);                   # Mesh spacing
  deltaT   = gMax*(deltaR)**2/maxnu       # Timestep
  nt       = int(t_end/deltaT);           # Number of time steps
  nStage   = len(gmlsrk);                 # Number of RK stages

  # Local value of g = nu*deltaT/deltaR^2
  for i in range(nr):
    if (r[i] < r1):
      g[i] = nu[i]*deltaT/(deltaR**2);
    elif (r[i] < r2):
      g[i] = nu[i]*deltaT/(deltaR**2);
    else:
      g[i] = nu[i]*deltaT/(deltaR**2);




  ani_skip   = int(((1/f_freq)/deltaT)/ani_count) # Time step skip for animated plots
  ani_first  = 1;

  # Plotting arrays
  plot_t      = np.array([0])                  # Plotting times
  plot_ubd    = np.array([0]); plot_ubd=u_amb  # Temperature at boundary vs time
  plot_ucl    = np.array([0]); plot_ucl=u_amb  # Temperature at centerline vs time
  ani_t       = np.array([0])                  # Animation times
  ani_u       = np.matrix(u_n)                 # Temperature vs radius



  #------------------------------------------------------
  #  March for nt steps, while saving solution
  #------------------------------------------------------
  for n in range(nt):

    #-------------------------------------------------------
    # Advance one step in time using a multi-stage RK method
    #-------------------------------------------------------
    for st in range(nStage):  

      # Evaluation time for this stage
      tst = t+deltaT*gmlsrk[st];      

      # Initial u vector for this stage
      if (st==0):                     
        u_st=np.copy(u_n)
      else:
        u_st=np.copy(u_np1)

      # Update the interior values 
      for i in range(1, nr-1):
        dudtDT = g[i]*(deltaR/(2*r[i])*(u_st[i+1]-u_st[i-1])+(u_st[i+1]-2*u_st[i]+u_st[i-1]));
        u_np1[i] =  u_n[i] + gmlsrk[st]*dudtDT;

      # Update the boundary values  *** MODIFY CODE HERE ****
      u_np1[0]  = u_amb;
      u_np1[-1] = u_amb + f_amp*np.sin(2*np.pi*f_freq*tst);   # External forcing


    #-------------------------------------------------------
    # Finish the time step
    #-------------------------------------------------------

    # Update to the new time and save the new solution
    t = t + deltaT;                                     
    u_n=np.copy(u_np1);                                


    # Check for maximum temperature deviation at centerline
    if (t >=t_amp ):                            
        u_clAmp = max(u_clAmp,abs(u_np1[cli]-u_amb))


    # Write to screen
    if (n % scr_skip) == 0 or n == nt-1:
        print(' n={:6d}'.format(n+1),'  t={:10.1f}'.format(t),'   u_clAmp=',u_clAmp, "r2 at this iteration", r2);

    # Save plot data
    if (n % plot_skip) == 0:
        plot_t   = np.append(plot_t, t)     
        plot_ubd = np.append(plot_ubd, u_np1[-1])    
        plot_ucl = np.append(plot_ucl, u_np1[cli])  

    # Save animation data
    if (t >=t_amp) and (n % ani_skip) == 0: 
        if (ani_first==1): 
          ani_t[0] = t     
          ani_u[0] = u_np1
          ani_first = 0
        else:  
          ani_t = np.append(ani_t, t)     
          ani_u = np.vstack((ani_u, u_np1))


  return (2.6e-4 - u_clAmp)


initial = [108e-3]

constraint_dict = {"type" : "ineq", 
              "fun": temperature_constraint}


bounds = [(21e-3, 195e-3)]  # r2 must lie between r1 and upper limit

solution = minimize(optimize_mass, x0=initial, bounds=bounds, constraints=[constraint_dict])



final_r2 = solution.x[0]

#------------------------------------------------------
# Mass per unit length 
#------------------------------------------------------
mass = np.pi*r1*r1*mat_rho[0] \
    + np.pi*(final_r2*final_r2-r1*r1)*mat_rho[1] \
    + np.pi*(r3*r3-final_r2*final_r2)*mat_rho[2]

#------------------------------------------------------
#%%  Print results
#------------------------------------------------------
print('')
print('---------------------------------------------------------------')
print('|  Results:                                                   |')
print('|                                                             |')
print('|  Mass per unit length: {:10f}'.format(mass),'kg/m                      |')
print('|                                                             |')
# print('|  Temperature amplitude at centreline: {:.4E}'.format(u_clAmp),'K          |')
print('|                 Final radius after optimization r=', final_r2, '         |')
print('---------------------------------------------------------------')
print('')

