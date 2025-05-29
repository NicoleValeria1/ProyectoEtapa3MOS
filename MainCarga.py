import pandas as pd
import numpy as np
from haversine import haversine
from geopy.distance import great_circle
import os
# Opcional: configuración del cliente de OpenRouteService (para vehículos terrestres)
# from openrouteservice import Client
# client = Client(key='TU_API_KEY')

# Parámetros generales
START_TIME = "07:45"  # Hora de inicio de operaciones (HH:MM)

# Carga de datos desde CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
pa1   = os.path.join(script_dir, 'clients_caso_3.csv')
pa2   = os.path.join(script_dir, 'depots.csv')
pa3   = os.path.join(script_dir, 'vehicles_caso_3.csv')
clients_df = pd.read_csv(pa1)
depots_df = pd.read_csv(pa2)
vehicles_df = pd.read_csv(pa3)

# Número de nodos y clientes
demanda_clients = clients_df['Demand'].tolist()
n_clients = len(demanda_clients)
n_depots = len(depots_df)  # normalmente 1 centro de distribución

# Índices de nodos:
# Nodo 1..n_depots: depósitos; n_depots+1..n_depots+n_clients: clientes;
# nodos de reabastecimiento: copias del CD al final
n_initial = n_depots + n_clients
V = n_clients
n = n_initial + V
print("Número de nodos:", n_initial)

# Vehículos
Vehiculos = len(vehicles_df)
Nombre_vehiculo = vehicles_df['VehicleID'].astype(str).tolist()

# Nodos de reabastecimiento y nodo de inicio
nodo_reabastecimiento = list(range(n_initial + 1, n + 1))
nodo_inicio = 1  # primer depósito

# Construcción de demandas: CD (0), luego clientes, luego reabastecimiento (0)
demanda = [0] * n_depots + demanda_clients + [0] * V

# Coordenadas reales de nodos
def build_coordinates():
    coords = {}
    # Depósitos
    for idx, row in depots_df.iterrows():
        coords[idx + 1] = (row['Latitude'], row['Longitude'])
    # Clientes
    for i, row in clients_df.iterrows():
        coords[n_depots + i + 1] = (row['Latitude'], row['Longitude'])
    # Reabastecimiento (mismos que el CD)
    for i in range(1, V+1):
        coords[n_initial + i] = coords[1]
    return coords

coordenadas = build_coordinates()

# Variables de vehículos cargadas desde CSV; se asume que si 'Speed' no está vacío, es un dron
vehiculos_info = {}
for _, row in vehicles_df.iterrows():
    vid = str(row['VehicleID'])
    tipo = 'Camioneta'
    vehiculos_info[vid] = {
        'tipo': tipo,
        'capacidad': float(row['Capacity']),
        'rango': float(row['Range'])
    }

# Factores de ajuste por tipo de vehículo
ajustes_vehiculo = {
    'Camioneta': {'cost_factor': 1.2, 'time_factor': 0.9},
    'Dron': {'cost_factor': 0.85, 'time_factor': 1.2}
}

# Factores de clima (usar condiciones reales o simuladas según necesidad)
factores_clima = {
    'Normal': {'cost_factor': 1.0, 'time_factor': 1.0},
    'Lluvia': {'cost_factor': 1.1, 'time_factor': 1.3},
    'Nieve': {'cost_factor': 1.3, 'time_factor': 1.7},
    'Tormenta': {'cost_factor': 1.5, 'time_factor': 2.0}
}
Condicion_Climatica = 'Normal'
factor_clima_cost = factores_clima[Condicion_Climatica]['cost_factor']
factor_clima_time = factores_clima[Condicion_Climatica]['time_factor']

# Costos operativos fijos (COP por km o unidad correspondiente)
Pf = 15000  # Precio del combustible (COP por litro)
Ft = 5000   # Tarifa de flete (COP por km)
Cm = 700    # Costo de mantenimiento (COP por km)
Seguros = 300  # Costo de seguros por km (COP)
Peajes = 2000  # Costo de peajes por km (COP)
Salarios = 8000  # Costo de salarios de conductor por km (COP)

# Cálculo de distancia según tipo de vehículo
def calcular_distancia(i, j, tipo):
    coord_i = coordenadas[i]
    coord_j = coordenadas[j]
    if tipo == 'Dron':
        return haversine(coord_i, coord_j)
    else:
        # terrestre: usar OpenRouteService o geopy como fallback
        return great_circle(coord_i, coord_j).kilometers

# Inicialización de matrices: distancia, costo y tiempo
cost = np.zeros((n, n, Vehiculos))
tiempos_viaje = np.zeros((n, n, Vehiculos))
matriz_distancias = np.zeros((n, n, Vehiculos))

for k, vid in enumerate(Nombre_vehiculo):
    Speed = 80  # Ignorar cualquier valor anterior, usamos velocidad fija
    tipo = vehiculos_info[vid]['tipo']
    cap = vehiculos_info[vid]['capacidad']
    rng = vehiculos_info[vid]['rango']
    factor_cost = ajustes_vehiculo[tipo]['cost_factor'] * factor_clima_cost
    factor_time = ajustes_vehiculo[tipo]['time_factor'] * factor_clima_time

    for i in range(1, n+1):
        for j in range(1, n+1):
            if i != j:
                d = round(calcular_distancia(i, j, tipo), 2)
                matriz_distancias[i-1, j-1, k] = d
                # Cálculo de costo usando los valores fijos originales
                costo_unitario = Pf + Ft + Cm + Seguros + Peajes + Salarios
                cost[i-1, j-1, k] = d * costo_unitario * factor_cost
                # Cálculo de tiempo (en horas) usando velocidad fija
                velocidad = 80  # velocidad constante
                tiempos_viaje[i-1, j-1, k] = round((d / velocidad) * factor_time, 2)
                #if i <= n_initial and j <= n_initial:
                    #print("El tiempo de viaje entre el nodo", i, "y el nodo", j, "es:", round(tiempos_viaje[i-1, j-1, k]*60, 2), "minutos para el vehículo", vid)
                    #print("La distancia entre el nodo", i, "y el nodo", j, "es:", d, "km para el vehículo", vid)

# Convertir matrices a listas para compatibilidad
cost = cost.tolist()
tiempos_viaje = tiempos_viaje.tolist()
matriz_distancias = matriz_distancias.tolist()

# Capacidades y rangos útiles por vehículo
Capacidad_util = {k+1: vehiculos_info[vid]['capacidad'] for k, vid in enumerate(Nombre_vehiculo)}
Distancia_util = {k+1: vehiculos_info[vid]['rango'] for k, vid in enumerate(Nombre_vehiculo)}

ventanas_tiempo = {}
for node in range(1, n + 1):
    ventanas_tiempo[node] = (0, 1440)  # ventana abierta todo el día (en minutos)

print(ventanas_tiempo)

# Verificación final
def verify_load():
    try:
        assert len(demanda) == n, "Vector de demanda con longitud incorrecta"
        assert cost and tiempos_viaje and matriz_distancias, "Matrices no inicializadas"
        assert len(ventanas_tiempo) == n, "Ventanas de tiempo faltantes"
        print("✅ Check: todo fue cargado e inicializado correctamente.")
    except AssertionError as e:
        print(f"❌ Error en verificación: {e}")

if __name__ == "__main__":
    verify_load()

