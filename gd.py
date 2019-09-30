import numpy as np 
 #The class for Gradient Decient (includes vanilla_gd, momentum, nag)
class gd_pv_2d:
    
    def __init__(self, fn_loss, fn_grad1, fn_grad2):
        self.fn_loss = fn_loss
        self.fn_grad1 = fn_grad1
        self.fn_grad2 = fn_grad2
        
    def vanilla_gd(self, x1_init,x2_init, n_iter, eta, tol):
        #initialize x1, x2, loss_path, x1_path, x2_path
        x1 = x1_init  
        x2 = x2_init
        
        loss_path = []
        x1_path = []
        x2_path = []
        
        #assign first values to paths 
        x1_path.append(x1) 
        g1 = self.fn_grad1(x1,x2)
        x2_path.append(x2)
        g2 = self.fn_grad2(x1,x2)
        loss_this = self.fn_loss(x1,x2)
        loss_path.append(loss_this)
        
        
        g_sum = g1**2 +g2**2   
        # the object is to minmise the sum of squared g1 and g2
        for i in range(n_iter):
            g1 = self.fn_grad1(x1,x2)
            g2 = self.fn_grad1(x1,x2)
            g_sum = g1**2 +g2**2 
            if g_sum < tol or np.isnan(g_sum):
                break
          
            x1 += -eta*g1
            x1_path.append(x1)
            
            x2 += -eta*g2
            x2_path.append(x2)
                        
            loss_this = self.fn_loss(x1,x2)
            loss_path.append(loss_this)
            
        if np.isnan(g_sum):
            print('Exploded')
        elif g_sum > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by x1 = {} ,x2 = {}'.format(i, loss_this, x1,x2))
        self.loss_path = np.array(loss_path)
        self.x1_path = np.array(x1_path)
        self.x2_path = np.array(x2_path)
    
    #-----
    def momentum(self, x1_init,x2_init, n_iter, eta, tol, alpha):
        #initialize x1, x2, loss_path, x1_path, x2_path
        x1 = x1_init
        x2 = x2_init
        loss_path = []
        x1_path = []
        x2_path = []
         #assign first values to paths
        x1_path.append(x1)
        g1 = self.fn_grad1(x1,x2)
        
        x2_path.append(x2)
        g2 = self.fn_grad2(x1,x2)
        
        loss_this = self.fn_loss(x1,x2)
        loss_path.append(loss_this)
        
        g_sum = g1**2 +g2**2
        
        vt1=0
        vt2=0

        for i in range(n_iter):
            g1 = self.fn_grad1(x1,x2)
            g2 = self.fn_grad1(x1,x2)
            g_sum = g1**2 +g2**2 
            if g_sum < tol or np.isnan(g_sum):
                break

            vt1 = alpha * vt1 + eta * g1
            x1 += -vt1
            x1_path.append(x1)
            
            vt2 = alpha * vt2 + eta * g2
            x2 += -vt2
            x2_path.append(x2)
                        
            loss_this = self.fn_loss(x1,x2)
            loss_path.append(loss_this)

        if np.isnan(g_sum):
            print('Exploded')
        elif g_sum > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by x1 = {} ,x2 = {}'.format(i, loss_this, x1,x2))
        self.loss_path = np.array(loss_path)
        self.x1_path = np.array(x1_path)
        self.x2_path = np.array(x2_path)
        
    #------------------

    def nag(self, x1_init,x2_init, n_iter, eta, tol, alpha):
            
            #initialize x1, x2, loss_path, x1_path, x2_path
            x1 = x1_init
            x2 = x2_init
            loss_path = []
            x1_path = []
            x2_path = []
            #assign first values to paths
            x1_path.append(x1)
            g1 = self.fn_grad1(x1,x2)
            x2_path.append(x2)
            g2 = self.fn_grad2(x1,x2)
        
            loss_this = self.fn_loss(x1,x2)
            loss_path.append(loss_this)
        
            g_sum = g1**2 +g2**2
        
            vt1=0
            vt2=0

            for i in range(n_iter):
                # i starts from 0 so add 1
                # The formula for mu was mentioned by David Barber UCL as being Nesterovs suggestion
                mu = 1 - 3 / (i + 1 + 5) 
                g1 = self.fn_grad1(x1 - mu*vt1,x2 - mu*vt2)
                g2 = self.fn_grad2(x1 - mu*vt1,x2 - mu*vt2)
                g_sum = g1**2 +g2**2
                if g_sum < tol or np.isnan(g_sum):
                    break

                vt1 = alpha * vt1 + eta * g1
                x1 += -vt1
                x1_path.append(x1)
            
                vt2 = alpha * vt2 + eta * g2
                x2 += -vt2
                x2_path.append(x2)
                        
                loss_this = self.fn_loss(x1,x2)
                loss_path.append(loss_this)

            if np.isnan(g_sum):
                print('Exploded')
            elif g_sum > tol:
                print('Did not converge')
            else:
                print('Converged in {} steps.  Loss fn {} achieved by x1 = {} ,x2 = {}'.format(i, loss_this,                 x1,x2))
            self.loss_path = np.array(loss_path)
            self.x1_path = np.array(x1_path)
            self.x2_path = np.array(x2_path)