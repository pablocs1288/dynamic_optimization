##### Inicialización de datos #####
# dados W's
W = np.array([
            [[0,0.4,0.3],
              [0.4, 0 , 0.1],
              [0.3, 0.1, 0]],
         [[0,0.5,0.4],
          [0.5, 0 , 0.2],
          [0.4, 0.2, 0]],
         [[0,0.6,0.1],
          [0.6, 0 , 0.15],
          [0.1, 0.15, 0]]])

#                    k=0          k=1  k=2
#s_example = np.array([[[0.03,0.027,0.01],  [], [], []])
#s_example =  np.array([[0.03,0.027,0.01]])

#                    k=0                                   k=1                                    k=2
#               n_j=1, n_j=2, n_j=3
#s_k = np.array([[[0.03,0.027,0.01],  [sub_gradite(x_j_1),sub_gradite(x_j_1),sub_gradite(x_j_1)], [], []])
s_k= [[0.6, 0.3, 0.5]] # acumuladores, que son listas!
a = [[0.01,0.03,0.12]] # -> [[0.01,0.03,0.12], [0.02,0.01,0.12432423]]  # acumuladores, que son listas!
pen = 0.75

#                    k=0          k=1  k=2
#x = np.array([[0.3,0.34,0.45],  [], [], []])
#            n_i=1, n_i=2, n_i=3
x = [[0.3,0.34,0.45]] # -> [[0.3,0.34,0.45], [0.2,0.1,0.65454],...] # acumuladores, que son listas!
# en x se acumulan, el x_nodos es lo que va dentro


##### implementación algoritmo ######


#y = 0.4 + 0.8*x + 0.57*x**2 + 0.54*|x| # sub_gradiente calcula dy/dx
def sub_gradiente(x):
    return 0.8 + 2*0.57*x

# implementacion del algoritmo
# 3 veces
for k in range(1,4): # cantidad de pasos que UD quiere que el algoritmo corra, o a cantidad de pasos que cada agente va a dar, K
    # second term
    #sum_nodos = np.dot(w_k_minus_1, x[k-1].reshape(3,1)) # producto de matrices
    
    # second term
    # para cada nodo j
    x_nodos = [] # sirven poara actualziar
    s_k_temp = []# sirven poara actualziar
    a_temp = []# sirven poara actualziar
    for i in range(3):  # la cantidad de nodos
        # calculando x(k) para el nodo i
        sum_nodos_j = np.dot(W[k-1][i], x[k-1])
        #        SUM                     alpha*s (subgraditne)
        x_k_i = np.dot(W[k-1][i], x[k-1]) - np.multiply(a[k-1][i], s_k[k-1][i])
        # aqui irian  las formulas, por ejemplo, la de EXTRA
        # x_k_i = x[k-1]+np.dot(W[k-1][i], x[k-1])-np.dot(W[k-2][i], x[k-2])-a[k-1]*(sub_gradiente(x[k-1]) - sub_gradiente(x[k-2]))
        x_nodos.append(x_k_i)
        
        # actaulizar para la próxima iteración
        #s_k_j_presente = s_k[k-1][j]*0.89 # el cálculo de mi funcion obejtivo para el nodo j en i
        s_k_i_presente = (x_k_i)
        s_k_temp.append(s_k_i_presente) # esto es un proceso más complejo, pero vamos a supenr quee stos son los sub-grads actualziadops
        a_temp.append(a[k-1][i]*pen) # el paso alpha para el nodo j se hace más pequeño a cada iteración
    # actualizacion general
    s_k.append(s_k_temp)
    a.append(a_temp)
    x.append(x_nodos)






###########################################
############ Material extra ###############
###########################################

#y = 0.4 + 0.8*x + 0.57*x**2 + 0.54*|x| # dy/dx
def sub_gradiente_ejemplo(x):
    if x == 0:
        # Considerar los intervales del sub-gradiente que ya calculo en papel y lapiz. ud ya deveria sabe rlos p[untos donde la funcion no es derivable
        return 1 # este valor tiene que estar de acuerdo al valor que UD ESCOGIO Dentro del intervalo del sub-gradiente!
    else:
        grad = 0.8 + 2*0.57*x
    return grad

# ejemplo de la FORMULA (siguiendo la misma lógica de arriba) de  EXTRA (exact first order algorithm)
# x[k-1]+np.dot(W[k-1][i], x[k-1])-np.dot(W[k-2][i], x[k-2])-a[k-1]*(sub_gradiente(x[k-1]) - sub_gradiente(x[k-2]))


# gradiente multivariado de ejemplo
def grad_ejemplo_multivariado(px_i, py_i,z_hat_,Z):
    z_hat_ = get_z(px_i, py_i, a, b1, b2, c11, c22, c33)
    dj_dz = 2*(z_hat_-Z)
    dj_da = dj_dz*1
    dj_db1 = dj_dz*px_i
    dj_db2 = dj_dz*py_i
    dj_dc11 = dj_dz*(px_i**2)
    dj_dc22 = dj_dz*(py_i**2)
    dj_dc33 = dj_dz*(px_i*py_i)

    return np.array([dj_dz,...,]) # esto no funciona, es solo un ejemplo didáctico




# ejemplos de listas!

# iteracion 1
lista_nodos = [
    ['Diego', 'Ana La loca', 'Gloria', 'Alby', 'Pablo']] # -> k = 0  

# iteracion 2
lista_familiasdiego = [
    ['Diego', 'Ana La loca', 'Gloria', 'Alby', 'Pablo'], # -> k = 0 
    ['Matilde', 'D. Alexander', 'Maria Gabriela', 'Gloria', 'Fernando']] # -> k = 1  

# imprimir Ana la loca

lista_familiasdiego[0][1] # !
