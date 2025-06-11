#=================================================================
# AE2220-II Computational Modelling.
# Heat equation in cylindrical coordinates solver
# RKLS in time, 2nd order central 
#=================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as pltani


#------------------------------------------------------
# Simulation parameters 
#------------------------------------------------------
r1         = 30e-3;              # Radius of glass (fixed)
r2         = 150e-3;             # Outer radius of green material (15e-3 < r2 <200e-3)
f_amp      = 3.3e-4;             # External forcing amplitude
f_freq     = 1e-4;               # External forcing frequency
cli        = 1;                  # Centreline grid index
u_amb      = 284;                # Ambient temperature
t_amp      = 5e4;                # Time after which amplitude is measured
t_end      = t_amp+2/f_freq;     # Simulation end time (2 periods after t_amp)


#--------------------------------------------------------------------
# Material properties: nu=diffusivity
#--------------------------------------------------------------------
nu_mat1   = 4.*10**(-7);
nu_mat2   = 8.*10**(-6);


#------------------------------------------------------
# Discretisation parameters  
#------------------------------------------------------
nr         = 51                            # Number of mesh points (min 50)
gMax       = 0.3;                          # Max nu*Delta t/ Delta r^2
gmlsrk     = np.array([0.1, 0.2, 0.5, 1])  # a, b, c, d for LSRK

#------------------------------------------------------
#  Plotting options 
#------------------------------------------------------
plots      = 1;         # Make plots if not zero
plot_skip  = 20;        # Time step skip for static plots
scr_skip   = 500;       # Time step skip for screen output
ani_count  = 10;        # Number of images per period in animation
ani_dint   = 100;       # Animation display interval [ms]



#------------------------------------------------------
# Make mesh and initialise solution vectors 
#------------------------------------------------------
r      = np.linspace(0, r2,  nr)     # Mesh coordinate vector
nu     = np.linspace(0, 1.0, nr)     # Diffusivity vector
g      = np.linspace(0, 1.0, nr)     # Local nu*deltaT/deltaX^2
u_n    = np.ones(nr)*u_amb;          # Temperature at time level n
u_st   = np.ones(nr)*u_amb;          # Temperature for RK stage
u_np1  = np.ones(nr)*u_amb;          # Temperature at time level n+1              


#------------------------------------------------------
# Set local diffusivity values
#------------------------------------------------------
for i in range(nr):
  if (r[i] < r1):
    nu[i]  = nu_mat1;
  else:
    nu[i]  = nu_mat2;


#------------------------------------------------------
# Compute time march parameters
#------------------------------------------------------
maxnu    = max(nu);                    # Max diffusivity encountered               
deltaR   = 1./(nr-1);                  # Mesh spacing
deltaT   = gMax*(deltaR)**2/maxnu      # Timestep
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



#------------------------------------------------------
# Prepare for time march
#------------------------------------------------------
t          = 0;                               # Initial time
u_clAmp    = 0;                               # Initial centerline temperature amplitude
ani_skip   = int(((1/f_freq)/deltaT)/ani_count) # Time step skip for animated plots
ani_first  = 1;

print("ani_skip",ani_skip)

# Plotting arrays
plot_t      = np.array([0])                  # Plotting times
plot_ubd    = np.array([0]); plot_ubd=u_amb  # Temperature at boundary vs time
plot_ucl    = np.array([0]); plot_ucl=u_amb  # Temperature at centerline vs time
ani_t       = np.array([0])                  # Animation times
ani_u       = np.matrix(u_n)                 # Temperature vs radius



print('')
print('---------------------------------------------------------------')
print('|  Marching for',nt,'time steps with a',nStage,'stage RKLS scheme   |')
print('---------------------------------------------------------------')
print('')



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
      u_st=u_n;   
    else:
      u_st=u_np1; 

    # Update the interior values 
    for i in range(1, nr-1):
      dudtDT = g[i]*(deltaR/(2*r[i])*(u_st[i+1]-u_st[i-1])+(u_st[i+1]-2*u_st[i]+u_st[i-1]));
      u_np1[i] =  u_n[i] + gmlsrk[st]*dudtDT;

    # Update the boundary values  *** MODIFY CODE HERE ****
    u_np1[0]  = u_amb                                       # Centerline 
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
      print('   n={:6d}'.format(n+1),'  t={:10.1f}'.format(t),'   u_clAmp=',u_clAmp);

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





#------------------------------------------------------
#  Print results
#------------------------------------------------------
print('')
print('---------------------------------------------------------------')
print('|  Results:                                                   |')
print('|                                                             |')
print('|  Temperature amplitude at centreline: {:.2E}'.format(u_clAmp),'K            |')
print('---------------------------------------------------------------')
print('')


#------------------------------------------------------
#  Plot results
#------------------------------------------------------
if plots != 0:


 fig = plt.figure(figsize=(10,10))

 ax1  = fig.add_subplot(211)
 ax1.grid();
 plt.plot(plot_t/10**(4),plot_ubd,label="u external")
 plt.plot(plot_t/10**(4),plot_ucl,label="u centreline")
 plt.title('Temperature history')
 plt.xlabel('tx10^4')
 plt.ylabel('u(t)')
 plt.legend()

 ax2 = fig.add_subplot(212)
 ax2.grid();
 ax2.set_ylim([u_amb-f_amp, u_amb+f_amp])
 plt.title('Temperature vs radius')
 plt.ylabel('u(r)')
 plt.xlabel('r')
 plt.axvspan(xmin=0,  xmax=r1, color='#1f77b4',alpha=0.1)
 plt.axvspan(xmin=r1, xmax=r2, color='gold',   alpha=0.1)
 udist = ax2.plot(r, r)[0]
 time  = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)


 def init():
     udist.set_ydata(np.ma.array(r,mask=True))
     time.set_text('')
     return udist, time

 def step(i):
     udist.set_ydata(ani_u[i])
     time.set_text('time = %.3fs'%(ani_t[i]))
     return udist, time
    
 ani = pltani.FuncAnimation(fig, step, ani_u.shape[0],init_func=init,
                           interval=ani_dint, blit=True)

 plt.show()
