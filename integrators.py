import numpy as np
import matplotlib.pyplot as plt

def Euler(dt, f, t, y, args):
    return y + f(t,y,*args) * dt 

def EulerCromer(dt, f, t, y, args):
    y_end = y + f(t,y,*args) * dt
    return y + f(t+dt, y_end, *args) * dt

def EulerRichardson(dt, f, t, y, args):
    y_mid = y + f(t,y,*args) * dt/2
    return y + f(t+dt/2, y_mid, *args) * dt


def solve_ode(f,tspan, y0, method = Euler, args=(), **options):
    """
    Given a function f that returns derivatives,
    dy / dt = f(t, y)
    and an inital state:
    y(tspan[0]) = y0
    
    This function will return the set of intermediate states of y
    from t0 (tspan[0]) to tf (tspan[1])
    
    
    
    The function is called as follows:
    
    INPUTS 
    
    f - the function handle to the function that returns derivatives of the 
        vector y at time t. The function can also accept parameters that are
        passed via *args, eg f(t,y,g) could accept the acceleration due to gravity.
        
    tspan - a indexed data type that has [t0 tf] as its two members. 
            t0 is the initial time
            tf is the final time
    
    y0 - The initial state of the system, must be passed as a numpy array.
    
    method - The method of integrating the ODEs. This week will be one of Euler, 
             Euler-Cromer, or Euler-Richardson
    
    *args - a tuple containing as many additional parameters as you would like for 
            the function handle f.
    
    **options - a dictionary containing all the keywords that might be used to control
                function behavior. For now, there is only one:
                
                first_step - the initial time step for the simulation.
    
    
    OUTPUTS
    
    t,y
    
    The returned states will be in the form of a numpy array
    t containing the times the ODEs were solved at and an array
    y with shape tsteps,N_y where tsteps is the number of steps 
    and N_y is the number of equations. Observe this makes plotting simple:
    
    plt.plot(t,y[:,0])
    
    would plot positions.
    
    """
    
    t0 = tspan[0]
    tf = tspan[1]
    y = [y0]
    t = [t0]
    dt = options.get('first_step')#,0.1)
    
    while t[-1]<tf:
        y.append(method(dt, f, t[-1], y[-1], args))
        t.append(t[-1] + dt)
    
    return np.array(t), np.array(y)