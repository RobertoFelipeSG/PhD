import numpy as np
import scipy as sp

def arpymorf (X,zz,Nr,Nl,p):

    #Init
    #x = areg
    #print(x.shape)
    #Nr = 1
    #Nl = df.shape[0]
    #p = 1
    x = np.asarray(X)
    x = np.reshape(x,(zz,1))
    R0  = np.zeros((zz,zz))
    pf  = R0
    pb  = R0
    pfb = R0
    En  = R0
    ap = np.zeros((zz,zz,1))
    bp = np.zeros((zz,zz,1))

    for i in range(Nr):
        En = En + np.dot(x[(i)*Nl:(i+1)*Nl,:], np.transpose(x[(i)*Nl:(i+1)*Nl,:]))
        ap[:,:,0] = ap[:,:,0] + np.dot((np.transpose(x[(i)*Nl+2:(i+1)*Nl,:])), x[(i)*Nl+2:(i+1)*Nl,:])        
        bp[:,:,0] = bp[:,:,0] + np.dot((np.transpose(x[(i)*Nl+1:(i+1)*Nl-1,:])), x[(i)*Nl+1:(i+1)*Nl-1,:])

    ap[:,:,0] = np.linalg.inv(np.linalg.cholesky(ap[:,:,0]/Nr*(Nl-1))) 
    bp[:,:,0] = np.linalg.inv(np.linalg.cholesky(bp[:,:,0]/Nr*(Nl-1))) 

    for i in range(Nr):
        efp = np.dot(ap[:,:,0], (x[(i)*Nl+1:(i+1)*Nl, :]).T)     ### What the hell is going on here?  
        ebp = np.dot(bp[:,:,0], (x[(i)*Nl+0:(i+1)*Nl-1, :]).T)
        pf = pf + np.matmul(efp, efp.T) 
        pb = pb + np.matmul(ebp, ebp.T)
        pfb = pfb + np.dot(efp, ebp.T)
        scipy.io.savemat('efp.mat', {'efp':efp})
        #print(pf)
        #print(pb)

    En = (np.linalg.cholesky(En/df.shape[0])) # Covariance of the noise
    #print(En)

    # Initial output variables
    #coeff = []; #  Coefficient matrices of the AR model
    kr=[];      # reflection coefficients
    a=np.zeros((2,2,2));
    b=np.zeros((2,2,2));

    for m in range(p):
        #m = m+1
        #print(m)
        # Calculate the next order reflection (parcor) coefficient
        ck = np.linalg.inv((np.linalg.cholesky(pf)).T) * pfb * np.linalg.inv(np.linalg.cholesky(pb))
        #print(ck)
        kr=[kr,ck]
        # Update the forward and backward prediction errors
        ef = np.identity(df.shape[1])- ck*ck.T
        eb = np.identity(df.shape[1])- ck.T*ck
        #print(ef)
        #print(eb)

        # Update the prediction error
        En = En * np.linalg.cholesky(ef).T
        E = (ef+eb)/2  

        # Update the coefficients of the forward and backward prediction errors
        ap1 = np.zeros((df.shape[1],df.shape[1],1))
        bp1 = np.zeros((df.shape[1],df.shape[1],1))
        ap = np.concatenate((ap, ap1), axis=2)
        bp = np.concatenate((bp, ap1), axis=2)
        pf = np.zeros(df.shape[1])
        pb = np.zeros(df.shape[1])
        pfb = np.zeros(df.shape[1])
        #print(ap[:,:,1])
        #print(bp[:,:,1])

        for i in range(m+2):  
            #print(i)
            a[:,:,i] = np.dot(np.linalg.inv((np.linalg.cholesky(ef)).T), ap[:,:,i]-ck*bp[:,:,m+2-(i+1)])
            b[:,:,i] = np.dot(np.linalg.inv((np.linalg.cholesky(eb)).T), bp[:,:,i]-ck.T*ap[:,:,m+2-(i+1)])
            #print(a[:,:,1])
            #print(b[:,:,1])

        for k in range(Nr):
            efp = np.zeros((df.shape[1],Nl-m-2))
            ebp = np.zeros((df.shape[1],Nl-m-2))
            #print(efp.shape)

            for i in range(m+2):
                #print(i)
                k1 = (m+1)+2-(i+1)+(k+1-1)*Nl+1
                #print(k1)
                k2 = Nl-(i+1)+1+(k+1-1)*Nl
                #print(k2)
                efp = efp + np.dot(a[:,:,i-1], x[k1-1:k2,:].T)
                ebp = ebp + np.dot(b[:,:,m+1-i], x[k1-2:k2-1,:].T)

            pf = pf + np.dot(efp, efp.T)
            pb = pb + np.dot(ebp, ebp.T)
            pfb = pfb + np.dot(efp, ebp.T)

        ap = a
        bp = b  

    for j in range(p):
        #print(j)
        coeff = np.dot(np.linalg.inv(a[:,:,0]), a[:,:,j+1])
        print(coeff)
        
    e = np.dot(En, np.linalg.inv(En))
        
    return coeff, e