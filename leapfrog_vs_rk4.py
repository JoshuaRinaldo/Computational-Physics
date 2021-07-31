# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:35:38 2021

@author: joshu
"""
import numpy as np  
import matplotlib.pyplot as plt  
import scipy.optimize as opt
import matplotlib 



#Question 2 works much better. The animations will close on their own. I set the number of steps per frame to be high 
#because I am impateint, so it won't take you too long to watch either.
one = True
two = True
three = True
four = True




if one == True:
    
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
    matplotlib.rc('font', **font)

    def find_root(guess):
        def lambdafunc(lambda_):
            #our lambda function
            return lambda_**5*(m2+m3)+lambda_**4*(2*m2+3*m3)+lambda_**3*(m2+3*m3)+lambda_**2*(-3*m1-m2)+lambda_*(-3*m1-2*m2)+(-m1-m2)
        def lambdafunc_prime(lambda_):
            #the derivative of our lambda function, with respect to lambda
            return 5*lambda_**4*(m2+m3)+4*lambda_**3*(2*m2+3*m3)+3*lambda_**2*(m2+3*m3)+2*lambda_*(-3*m1-m2)+(-3*m1-2*m2)
        return opt.root_scalar(lambdafunc,x0=0.2,fprime=lambdafunc_prime,method='newton')
    
    #calculates a. I called it a_cubed but it is technically the cubic root of a. Just doesn't roll off the tongue as well!
    def a_cubed(lambda_,omega):
        return (omega**(-2)*(m2+m3-m1*(1+2*lambda_)/(lambda_**2*(1+lambda_)**2)))**(1/3)
    
    #calculates x0 values as outlined in question 2
    def gen_x0_for_question(lambda_,omega,a):
        x2 = 1/(omega**2*a**2)*(m1/lambda_**2-m3)
        x1 = x2 - lambda_*a
        x3 =-(m1*x1+m2*x2)/m3
        return x1,x2,x3
    
    def calculate_a(r_array):
        #calculates distances between bodies
        r12,r13,r23 = r_array[0]-r_array[1],r_array[0]-r_array[2],r_array[1]-r_array[2] 
    
        #calculates radius between bodies
        s12 = np.sqrt(r12[0]**2+r12[1]**2)
        s13 = np.sqrt(r13[0]**2+r13[1]**2)
        s23 = np.sqrt(r23[0]**2+r23[1]**2)
        
        #calculates accelerations a0 = -Gm1(x0-x1)/(r^3), G = 1
        a1 = -m2*(r12)/(s12**3) - m3*(r13)/(s13**3)
        a2 = -m1*(-r12)/(s12**3) - m3*(r23)/(s23**3)
        a3 = -m1*(-r13)/(s13**3) - m2*(-r23)/(s23**3)
        accelerations = [a1,a2,a3]
        return np.array(accelerations)
    
    #the dx/dt part of our ODE
    def calculate_v(velocities):
        return np.array(velocities)
    
    #our ode function
    def derivs(all_args,t): #all_args[0] = positions all_args[1] = velocity
        dy1 = calculate_a(all_args[0]) #velocity ODE
        dy0 = calculate_v(all_args[1]) #position ODE
        return np.array([dy0,dy1])
    def calc_energy_bodyproblem(r_array,v_array):
        #calculates distances between bodies
        r12,r13,r23 = r_array[0]-r_array[1],r_array[0]-r_array[2],r_array[1]-r_array[2] 
    
        #calculates radius between bodies
        s12 = np.sqrt(r12[0]**2+r12[1]**2)
        s13 = np.sqrt(r13[0]**2+r13[1]**2)
        s23 = np.sqrt(r23[0]**2+r23[1]**2)
        
        v1_squared = v_array[0][0]**2+v_array[0][1]**2
        v2_squared = v_array[1][0]**2+v_array[1][1]**2
        v3_squared = v_array[2][0]**2+v_array[2][1]**2
        #calculates potential energy
        potential_energy = m1*m2/s12 + m1*m3/s13 + m2*m3/s23
        #calculates kinetic energy
        kinetic_energy = 0.5*(m1*v1_squared + m2*v2_squared + m3*v3_squared)
        #calculates total energy
        mechanical_energy = potential_energy+kinetic_energy
        return mechanical_energy
    
    #our farmiliar RK4 method
    def RK4(f,y,t,dt):
        K1 = np.asarray((dt*f(y,t)))
        K2 = np.asarray((dt*f(y+K1/2,t+dt/2)))
        K3 = np.asarray((dt*f(y + K2/2,t+dt/2)))
        K4 = np.asarray((dt*f(y+K3,t+dt)))
        return y + (K1+2*K2+2*K3+K4)/6
    
    #the leapfrog method
    def leapfrog_(h,arg_array):
        h_halved = h/2.0
        position_10ver2 = arg_array[0] + np.array(arg_array[1])*h_halved 
        
        v1 = arg_array[1] + h*np.array(calculate_a(arg_array[0]))
        
        position_1 = position_10ver2 + h_halved*v1
        
        return position_1,v1
    
    def run_animation(start,stop,stepsize,steps_per_frame,rk4_odient_or_leapfrog,all_args,title,velocity_flip):
        fig,(ax_E,ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,3]})
        plt.subplots_adjust(hspace=0.6)
        ax.set_xlabel('$x$ (Plank Length)')     # add labels
        ax.set_ylabel('$y$ (Plank Length)')
        ax.set_title('Body Positions'+title)
        #labels for our Energy graph
        ax_E.set_xlabel('$t$ (Plank time)')    # add labels
        ax_E.set_ylabel('$E$ (J)')
        ax_E.set_title('Energy Error')
        E_line, = ax_E.plot(1,1,"k--")
        #create our energy and time lists for later plotting
        time = []
        E = []
        #all of our dynamic parts of our position graph
        line1, = ax.plot( 1, 1,'ro', label = "m1", markersize=10)
        path1, = ax.plot( 1, 1,'r--', markersize=4)
        path1x = []
        path1y = []
        arrow1 = ax.quiver(1,1,1,1, color = "k")
        line2, = ax.plot( 1, 1,'bo', label = "m2", markersize=10)
        path2, = ax.plot( 1, 1,'b--', markersize=4)
        path2x = []
        path2y = []
        arrow2 = ax.quiver(1,1,1,1, color = "k")
        line3, = ax.plot( 1, 1,'go', label = "m3", markersize=10)
        path3, = ax.plot( 1, 1,'g--', markersize=4)
        path3x = []
        path3y = []
        arrow3 = ax.quiver(1,1,1,1, color = "k")
        c_o_m, = ax.plot(1,1,"ko",label = "centre of mass", markersize = 5)
        #creates the starting positions of our masses
        m1_start, = ax.plot( all_args[0][0][0], all_args[0][0][1],'ro', label = "m1 start", markersize=5)
        m2_start, = ax.plot( all_args[0][1][0], all_args[0][1][1],'bo', label = "m2 start", markersize=5)
        m3_start, = ax.plot( all_args[0][2][0], all_args[0][2][1],'go', label = "m3 start", markersize=5)
        
        stepcounter = 0
        tpause = 0.001 # delay within animation 
                     #(though the minimum depends on your specs)
        E_true = calc_energy_bodyproblem(all_args[0],all_args[1])
        
        x_arr = []
        y_arr = []
        
        if rk4_odient_or_leapfrog == "leapfrog":
            for n in np.arange(start,stop,stepsize):
                all_argszero,all_argsone = leapfrog_(stepsize,all_args)
                all_args[0] = all_argszero
                all_args[1] = all_argsone
                #checking if we are flipping velocities.
                if velocity_flip != 0:
                    if stepcounter == velocity_flip:
                        all_args[1] = -all_args[1] #flips velocity
                
                #plotting our animation
                if stepcounter%steps_per_frame == 0:
                    #adds to our path lines
                    path1x.append(all_args[0][0][0])
                    path1y.append(all_args[0][0][1])
                    path2x.append(all_args[0][1][0])
                    path2y.append(all_args[0][1][1])
                    path3x.append(all_args[0][2][0])
                    path3y.append(all_args[0][2][1])
                    #plots our paths
                    path1.set_xdata(path1x)
                    path1.set_ydata(path1y)
                    path2.set_xdata(path2x)
                    path2.set_ydata(path2y)
                    path3.set_xdata(path3x)
                    path3.set_ydata(path3y)
                    for x_y in all_argszero:
                        #appends values to our x and y array
                        x_arr.append(x_y[0])
                        y_arr.append(x_y[1])
                    #sets the ends of the axes to be the largest and smallest positions in x and y
                    ax.set_ylim(min(y_arr)-1,max(y_arr)+1)
                    ax.set_xlim(min(x_arr)-1,max(x_arr)+1)
                    
                    #calculates and appends energy to our line plot
                    E0 = calc_energy_bodyproblem(all_argszero,all_argsone)
                    E.append(E0-E_true)
                    time.append(n)
                    #plots line
                    E_line.set_data(time,E)
                    #updates line plot axes
                    ax_E.set_ylim(min(E),max(E))
                    ax_E.set_xlim(min(time),max(time))
                    #plots our planetary bodies and their velocities
                    line1.set_xdata(all_args[0][0][0])
                    line1.set_ydata(all_args[0][0][1])
                    arrow1.set_offsets([all_args[0][0][0],all_args[0][0][1]])
                    arrow1.set_UVC(all_args[1][0][0],all_args[1][0][1])
                    
                    line2.set_xdata(all_args[0][1][0])
                    line2.set_ydata(all_args[0][1][1])
                    arrow2.set_offsets([all_args[0][1][0],all_args[0][1][1]])
                    arrow2.set_UVC(all_args[1][1][0],all_args[1][1][1])
                    
                    line3.set_xdata(all_args[0][2][0])
                    line3.set_ydata(all_args[0][2][1])
                    arrow3.set_offsets([all_args[0][2][0],all_args[0][2][1]])
                    arrow3.set_UVC(all_args[1][2][0],all_args[1][2][1])
                    
                    #calculates and plots centre of momentum
                    com_x = m1*all_args[0][0][0]+m2*all_args[0][1][0]+m3*all_args[0][2][0]
                    com_y = m1*all_args[0][0][1]+m2*all_args[0][1][1]+m3*all_args[0][2][1]
                    c_o_m.set_xdata(com_x)
                    c_o_m.set_ydata(com_y)
                    
                    ax.legend(prop={"size":10})
                    plt.draw()
                    plt.pause(tpause) # pause to see animation as code v. fast
                stepcounter+=1
        if rk4_odient_or_leapfrog == "RK4":
            for n in np.arange(start,stop,stepsize):
                #applies the RK4 method
                all_argszero,all_argsone = RK4(derivs,all_args,n,stepsize)
                all_args[0] = all_argszero
                all_args[1] = all_argsone
                #checking if we are flipping velocities.
                if velocity_flip != 0:
                    if stepcounter == velocity_flip:
                        all_args[1] = -all_args[1] #flips velocity
                #plotting our animation
                if stepcounter%steps_per_frame == 0:
                    #adds to our paths
                    path1x.append(all_args[0][0][0])
                    path1y.append(all_args[0][0][1])
                    path2x.append(all_args[0][1][0])
                    path2y.append(all_args[0][1][1])
                    path3x.append(all_args[0][2][0])
                    path3y.append(all_args[0][2][1])
                    #plots our paths
                    path1.set_xdata(path1x)
                    path1.set_ydata(path1y)
                    path2.set_xdata(path2x)
                    path2.set_ydata(path2y)
                    path3.set_xdata(path3x)
                    path3.set_ydata(path3y)
                    #making dynamic x and y axes for our body positions
                    for x_y in all_argszero:
                        x_arr.append(x_y[0])
                        y_arr.append(x_y[1])
                    ax.set_ylim(min(y_arr),max(y_arr))
                    ax.set_xlim(min(x_arr),max(x_arr))
                    
                    E0 = calc_energy_bodyproblem(all_argszero,all_argsone)
                    E.append(E0-E_true)
                    time.append(n)
                    ax_E.set_ylim(min(E),max(E))
                    ax_E.set_xlim(min(time),max(time))
                    E_line.set_data(time,E)
                    
                    line1.set_xdata(all_args[0][0][0])
                    line1.set_ydata(all_args[0][0][1])
                    arrow1.set_offsets([all_args[0][0][0],all_args[0][0][1]])
                    arrow1.set_UVC(all_args[1][0][0],all_args[1][0][1])
                    
                    line2.set_xdata(all_args[0][1][0])
                    line2.set_ydata(all_args[0][1][1])
                    arrow2.set_offsets([all_args[0][1][0],all_args[0][1][1]])
                    arrow2.set_UVC(all_args[1][1][0],all_args[1][1][1])
                    
                    line3.set_xdata(all_args[0][2][0])
                    line3.set_ydata(all_args[0][2][1])
                    arrow3.set_offsets([all_args[0][2][0],all_args[0][2][1]])
                    arrow3.set_UVC(all_args[1][2][0],all_args[1][2][1])
                    
                    #calculates centre of mass
                    com_x = m1*all_args[0][0][0]+m2*all_args[0][1][0]+m3*all_args[0][2][0]
                    com_y = m1*all_args[0][0][1]+m2*all_args[0][1][1]+m3*all_args[0][2][1]
                    c_o_m.set_xdata(com_x)
                    c_o_m.set_ydata(com_y)
                    
                    ax.legend(prop={"size":10})
                    plt.draw()
                    plt.pause(tpause) # pause to see animation
                stepcounter+=1
        plt.close() #automatically closes the plot so the marker doesn't   
                    #have to deal with multiple windows of finished plots
        
    #now that we have finished defining our functions, we can move on to running the animations
    m1 = 1
    m2 = 2
    m3 = 3
    root = find_root(0.2)
    lambda_ =  root.root #gets lambda
    
    if two == True:
        a = a_cubed(lambda_,1)
        x1,x2,x3 = gen_x0_for_question(lambda_,1,a)
        #planets = [[[x1,y1],[x2,y1],[x3,y3]],[[vx1,vy1],[vx2,vy1],[vx3,vy3]]]
        planets = np.array([[[x1,0],[x2,0],[x3,0]],[[0,x1],[0,x2],[0,x3]]])
        #First we run the simulation using the leapfrog method
        run_animation(0,8*np.pi,0.001,100,"leapfrog",planets," Leapfrog - runtime = $4T_0$ - $\omega = 1$",0)
        #Then we run the simulation using the RK4 method
        planets = np.array([[[x1,0],[x2,0],[x3,0]],[[0,x1],[0,x2],[0,x3]]])
        run_animation(0,8*np.pi,0.001,100,"RK4",planets," RK4 - runtime = $4T_0$ - $\omega = 1$",0)
        
        
        #then we swtich omega to be equal to + delta
        delta = 1e-9
        a = a_cubed(lambda_,1+delta)
        x1,x2,x3 = gen_x0_for_question(lambda_,1+delta,a)
        planets = np.array([[[x1,0],[x2,0],[x3,0]],[[0,x1],[0,x2],[0,x3]]])
        #First we run the simulation using the leapfrog method
        run_animation(0,8*np.pi,0.001,100,"leapfrog",planets," Leapfrog - runtime = $4T_0$ - $\omega = 1+\delta$",0)
        #Then we run the simulation using the RK4 method
        planets = np.array([[[x1,0],[x2,0],[x3,0]],[[0,x1],[0,x2],[0,x3]]])
        run_animation(0,8*np.pi,0.001,100,"RK4",planets," RK4 - runtime = $4T_0$ - $\omega = 1+\delta$",0)
        
        #then we swtich omega to be equal to - delta
        a = a_cubed(lambda_,1-delta)
        x1,x2,x3 = gen_x0_for_question(lambda_,1-delta,a)
        planets = np.array([[[x1,0],[x2,0],[x3,0]],[[0,x1],[0,x2],[0,x3]]])
        #First we run the simulation using the leapfrog method
        planets = np.array([[[x1,0],[x2,0],[x3,0]],[[0,x1],[0,x2],[0,x3]]])
        run_animation(0,8*np.pi,0.001,100,"leapfrog",planets," Leapfrog - runtime = $4T_0$ - $\omega = 1-\delta$",0)
        #Then we run the simulation using the RK4 method
        planets = np.array([[[x1,0],[x2,0],[x3,0]],[[0,x1],[0,x2],[0,x3]]])
        run_animation(0,8*np.pi,0.001,100,"RK4",planets," RK4 - runtime = $4T_0$ - $\omega = 1-\delta$",0)
    
    
    
    
    
    #We now double our timescale to 16pi. Don't worry, I preset the step counter so it doesn't take twice as long
    if three == True:
        delta = 1e-9
        a = a_cubed(lambda_,1+delta)
        x1,x2,x3 = gen_x0_for_question(lambda_,1+delta,a)
        planets = np.array([[[x1,0],[x2,0],[x3,0]],[[0,x1],[0,x2],[0,x3]]])
        #First we run the simulation using the leapfrog method
        run_animation(0,16*np.pi,0.001,200,"leapfrog",planets," Leapfrog - runtime = $16T_0$ - $\omega = 1+\delta$, flipping velocities",20000)
        #Then we run the simulation using the RK4 method
        planets = np.array([[[x1,0],[x2,0],[x3,0]],[[0,x1],[0,x2],[0,x3]]])
        run_animation(0,16*np.pi,0.001,200,"RK4",planets," RK4 - runtime = $16T_0$ - $\omega = 1+\delta$, flipping velocities",20000)
        #it seems as though the RK4 method does well if the velocities are flipped. There is a peak in error, which is usually shortly
        #followed by many more periodic peaks in error, but when velocities are flipped, this goes away.
        
    if four == True:
        m1 = 1/3
        m2 = 1/3
        m3 = 1/3
        planets = np.array([[[-0.30805788,0],[0.15402894,-0.09324743],[0.15402894,0.09324743]],[[0,-1.015378093],[0.963502817,0.507689046],[-0.963502817,0.507689046]]])
        #First we run the simulation using the leapfrog method
        run_animation(0,3*2*np.pi/3.33,0.001,50,"leapfrog",planets," Leapfrog - runtime = $3T_0$ - d.i)",0)
        
        
        m1 = 1
        m2 = 1
        m3 = 1
        planets = np.array([[[0.97000436,-0.24308753],[-0.97000436,0.24308753],[0,0]],[[0.93240737/2.,0.86473146/2.],[0.93240737/2.,0.86473146/2.],[-0.93240737,-0.86473146]]])
        #First we run the simulation using the leapfrog method
        run_animation(0,3*2*np.pi/2.47,0.001,50,"leapfrog",planets," Leapfrog - runtime = $3T_0$ - d.ii",0)
        
        #the first set of initial conditions is clearly not stable. We will check the stability of the second
        #initial conditions over twice the period
        run_animation(0,6*2*np.pi/3.33,0.001,100,"leapfrog",planets," Leapfrog - runtime = $6T_0$ - d.ii",0)