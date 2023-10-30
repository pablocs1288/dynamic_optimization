# importaciones
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
Definición de métodos
"""

# Definición del método para la expresión de Z
def get_z(px_i, py_i, a, b_1, b_2, c_11, c_22, c_33):
    _z = a + b_1*px_i + b_2*py_i  + c_33*px_i*py_i + c_11*(px_i**2) + c_22*(py_i**2)
    return _z

# Implementación del algoritmo del gradiente descendente: Referencia, curso de Andrew NG de deep learning y https://www.youtube.com/watch?v=EfsjEOb596Q&t=836s
def gradient_descent(df, num_iterations, alpha = 0.001):

    # init coef. aleatoriamente
    s = np.random.uniform(-5,5,6)
    a = s[0]
    b1 = s[1]
    b2 = s[2]
    c11 = s[3]
    c22 = s[4]
    c33 = s[5]

    dj_da, dj_db1, dj_db2, dj_dc11, dj_dc22, dj_dc33 = 0,0,0,0,0,0
    for it in range(num_iterations):
        # derivatives
        for P in df[['Px', 'Py', 'Z']].values:
            px_i = P[0]
            py_i = P[1]
            Z = P[2]
            # delta calculus
            z_hat_ = get_z(px_i, py_i, a, b1, b2, c11, c22, c33)
            dj_dz = 2*(z_hat_-Z)
            dj_da += dj_dz*1
            dj_db1 += dj_dz*px_i
            dj_db2 += dj_dz*py_i
            dj_dc11 += dj_dz*(px_i**2)
            dj_dc22 += dj_dz*(py_i**2)
            dj_dc33 += dj_dz*(px_i*py_i)
        
        n = len(df)
        dj_da = dj_da/n
        dj_db1 = dj_db1/n
        dj_db2 = dj_db2/n
        dj_dc11 = dj_dc22/n
        dj_dc22 = dj_dc22/n
        dj_dc33 = dj_dc33/n
        
        # actualizaciones de los coeficientes conforme la dirección que indica cada derivada parcial
        a = a - alpha*dj_da
        b1 = b1 - alpha*dj_db1
        b2 = b2 - alpha*dj_db2
        c11 = c11 - alpha*dj_dc11
        c22 = c22 - alpha*dj_dc22
        c33 = c33 - alpha*dj_dc33

    # coeficientes
    return {'a':a, 'b1':b1, 'b2': b2, 'c11': c11, 'c22': c22, 'c33': c33}

# gráficos en 3d de la figura original (azul) y de la figura ajustada (rojo)
def graficar(Px, Py, Z_original, Z_hat, file_name='ajuste_plano.png', path = 'D:\optimizacion_dinamica\taller1\img'):
    # Gráfico de z_hat_ (función con los coeficientes calculados) vs z (variables del dataset)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(Px, Py, Z_original, c='blue', marker='o')
    ax.scatter(Px, Py, Z_hat, c='red', marker='o') 

    ax.set_xlabel('Px')
    ax.set_ylabel('Py')
    ax.set_zlabel('Z')
    plt.savefig(path+file_name, dpi=300, bbox_inches='tight')
    print('figura guardada en ',path+file_name)



"""
Ejucución del programa
"""

# lectura de los datos
df_ = pd.read_excel('D:\optimizacion_dinamica\taller1\datos\testdata.xlsx') # cambiar la direccción del archivo si es necesario
df_work = df_[1:][['Unnamed: 0','Unnamed: 1', 'Diego R.']]
df_work.columns = ['Px','Py','Z']

## figura con coeficientes aleatórios ##
s = np.random.uniform(-5,5,6)
a = s[0]
b1 = s[1]
b2 = s[2]
c11 = s[3]
c22 = s[4]
c33 = s[5]
z_hat_random = df_work.apply(lambda row: get_z(row['Px'], row['Py'], a, b1, b2, c11, c22, c33), axis = 1)
graficar(df_work['Px'], df_work['Py'], df_work['Z'],z_hat_random, file_name="ajuste_plano_coef_random.png", path='D:\optimizacion_dinamica\taller1\img')


## Gradiente descendente ##
# ejecución del algoritmo de gradiente descendiente, los coeficientes finales se guardan en el diccionario opt
opt= gradient_descent(df_work, 50000, 0.01)
# con los coeficientes calculados por la optimización, se calcula la proyección de la función z_hat_ 
z_hat_ = df_work.apply(lambda row: get_z(row['Px'], row['Py'], opt['a'], opt['b1'], opt['b2'], opt['c11'], opt['c22'], opt['c33']), axis = 1)
graficar(df_work['Px'], df_work['Py'], df_work['Z'], z_hat_, file_name="ajuste_plano_coef_con_gradiente_descendente.png", path='D:\optimizacion_dinamica\taller1\img')
print('Coeficientes después del gradiente:', opt)

