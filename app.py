# app.py
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
image_data = []  # Inicializar variable global
network_config = {}  # Variable global para almacenar la configuración de la red
trained_weights = []  # Variable global para almacenar los pesos entrenados

# Crear el directorio de carga si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('Main.html')

# Función para convertir la imagen en matriz 10x10, binarizar y calcular el vector suma
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  
    image = image.resize((10, 10), Image.Resampling.LANCZOS) 
    image_array = np.array(image)
    binary_image = np.where(image_array > 127, 1, 0)  
    vector_sum = np.sum(binary_image, axis=0) 
    return vector_sum, image_path

@app.route('/upload', methods=['POST'])
def upload_data():
    if 'files[]' not in request.files:
        return "No se subieron archivos"

    files = request.files.getlist('files[]')
    global image_data
    image_data = []  # Reiniciar datos
    for file in files:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        vector_sum, path = preprocess_image(image_path)
        image_data.append((vector_sum.tolist(), path))
    
    # Guardar vectores suma en un archivo CSV
    with open('vectors.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Vector', 'Image Path'])
        for vector, path in image_data:
            writer.writerow([vector, path])

    return render_template('Results.html', data=image_data, image_found=None)

@app.route('/find_image', methods=['POST'])
def find_image():
    input_vector = request.form['vector']
    input_vector = list(map(int, input_vector.split(',')))

    global image_data
    image_path = None
    for vector, path in image_data:
        if vector == input_vector:
            image_path = path
            break

    return render_template('Results.html', data=image_data, image_found=image_path)

@app.route('/config')
def config():
    global image_data
    num_patterns = len(image_data)
    num_inputs = len(image_data[0][0]) if num_patterns > 0 else 0
    num_neurons = num_inputs * 2
    
    # Calcular el coeficiente de vecindad
    distances = np.random.uniform(0, 1, (num_neurons, num_neurons))
    np.fill_diagonal(distances, 0)  # No hay distancia entre una neurona y sí misma
    neighborhood_coef = np.sum(distances) / (num_neurons * (num_neurons - 1))
    neighborhood_coef = round(neighborhood_coef, 2)  # Limitar a 2 decimales

    return render_template('Config.html', num_patterns=num_patterns, num_inputs=num_inputs,
                           num_neurons=num_neurons, iterations=100, neighborhood_coef=neighborhood_coef,
                           learning_rate=1)

@app.route('/set_config', methods=['POST'])
def set_config():
    global network_config
    num_patterns = request.form['num_patterns']
    num_inputs = request.form['num_inputs']
    num_neurons = int(request.form['num_neurons'])
    competition_type = request.form['competition_type']
    iterations = int(request.form['iterations'])
    neighborhood_coef = request.form['neighborhood_coef']
    learning_rate = request.form['learning_rate']

    # Inicializar pesos sinápticos
    weights = np.random.uniform(-0.1, 0.1, (int(num_inputs), num_neurons))

    network_config = {
        'num_patterns': num_patterns,
        'num_inputs': num_inputs,
        'num_neurons': num_neurons,
        'competition_type': competition_type,
        'iterations': iterations,
        'neighborhood_coef': neighborhood_coef,
        'learning_rate': learning_rate,
        'weights': weights.tolist()  # Convertir a lista para facilitar JSON
    }

    return render_template('Config.html', num_patterns=num_patterns, num_inputs=num_inputs,
                           num_neurons=num_neurons, iterations=iterations, neighborhood_coef=neighborhood_coef,
                           learning_rate=learning_rate, message="Configuración guardada y pesos inicializados.")

@app.route('/train', methods=['POST'])
def train():
    global network_config, trained_weights
    neighborhood_coef = float(network_config['neighborhood_coef'])
    num_patterns = int(network_config['num_patterns'])
    num_inputs = int(network_config['num_inputs'])
    num_neurons = int(network_config['num_neurons'])
    competition_type = network_config['competition_type']
    iterations = int(network_config['iterations'])
    learning_rate = 1.0
    weights = np.array(network_config['weights'])

    # Cargar y verificar los datos
    try:
        # Leer el archivo CSV completo primero para debugging
        with open('vectors.csv', 'r') as f:
            print("Contenido del archivo vectors.csv:")
            print(f.read())
        
        # Cargar los datos nuevamente para procesamiento
        data = []
        with open('vectors.csv', 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Saltar el encabezado
            for row in csv_reader:
                # Extraer solo los números del vector
                vector_str = row[0]  # Primera columna
                # Convertir string de lista a lista real
                vector = [float(x.strip()) for x in vector_str.strip('[]').split(',')]
                data.append(vector)
        
        data = np.array(data)
        print("Datos cargados:")
        print("Forma de los datos:", data.shape)
        print("Primeras filas:", data[:5])
        
    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")
        return "Error al cargar los datos"

    # Variables para seguimiento
    dm_values = []
    synaptic_weights = []

    print("Iniciando entrenamiento...")
    for iteration in range(1, iterations + 1):
        dm_sum = 0
        valid_patterns = 0

        for pattern in data:
            if np.any(np.isnan(pattern)):
                print(f"Patrón con NaN encontrado: {pattern}")
                continue

            # Calcular distancias para cada neurona
            distances = np.zeros(num_neurons)
            for i in range(num_neurons):
                distances[i] = np.sqrt(np.sum((pattern - weights[:, i]) ** 2))

            winner_idx = np.argmin(distances)
            current_distance = distances[winner_idx]
            
            # Actualizar suma DM y contador
            dm_sum += current_distance
            valid_patterns += 1

            # Actualizar pesos
            if competition_type == 'hard':
                weights[:, winner_idx] += learning_rate * (pattern - weights[:, winner_idx])
                print(f"Pesos después de actualizar (hard) - Neurona {winner_idx}: {weights[:, winner_idx]}")
            elif competition_type == 'soft':
                for i in range(num_neurons):
                    if i != winner_idx:
                        weights[:, i] += (learning_rate * neighborhood_coef) * (pattern - weights[:, i])
                weights[:, winner_idx] += learning_rate * (pattern - weights[:, winner_idx])
                print(f"Pesos después de actualizar (soft): {weights}")

        #print(f"Pesos después de actualizar (hard) - Neurona {winner_idx}: {weights[:, winner_idx]}")
        #print(f"Pesos después de actualizar (soft): {weights}")
        
        # Calcular DM para esta iteración
        if valid_patterns > 0:
            dm = dm_sum / valid_patterns
        else:
            dm = float('inf')
            print(f"Advertencia: No hay patrones válidos en la iteración {iteration}")

        print(f"Iteración {iteration}: DM = {dm}")
        dm_values.append(dm)
        synaptic_weights.append(weights.copy())

        # Actualizar tasa de aprendizaje
        learning_rate = 1.0 / (iteration + 1)

        # Graficar solo si hay valores válidos
        if not all(np.isinf(dm_values)):
            plt.figure(figsize=(10, 5))
            valid_dms = [x for x in dm_values if not np.isinf(x)]
            plt.plot(range(1, len(valid_dms) + 1), valid_dms, label='Distancia Media (D m)')
            plt.xlabel('Iteraciones')
            plt.ylabel('DM')
            plt.title('Evolución de DM durante el Entrenamiento')
            plt.legend()
            plt.grid(True)
            plt.savefig('static/dm_plot_final.png')
            plt.close()

        plt.figure(figsize=(10, 5))
        for i in range(weights.shape[1]):
            plt.scatter(range(weights.shape[0]), weights[:, i], label=f'Neurona {i}')
        plt.xlabel('Entradas')
        plt.ylabel('Pesos')
        plt.title('Evolución de los Pesos Sinápticos durante el Entrenamiento')
        plt.legend()
        plt.grid(True)
        plt.savefig('static/weights_plot_final.png')
        plt.close()
        

    trained_weights = weights.tolist()

    # Guardar resultados
    with open('dm_values.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["DM"])
        writer.writerows([[dm] for dm in dm_values])

    with open('weights.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'Neurona {i}' for i in range(weights.shape[1])])
        writer.writerows(weights.T)

    print("Pesos después del entrenamiento:")
    print("Shape:", np.array(trained_weights).shape)
    print("¿Contiene NaN?:", np.any(np.isnan(np.array(trained_weights))))

    return redirect(url_for('simulate', train_message="Entrenamiento completado y pesos actualizados."))


@app.route('/simulate')
def simulate():
    return render_template('Simulate.html', message="Entrenamiento completado. Introduce nuevos patrones de entrada para probar la red.",
                           dm_plot='static/dm_plot_final.png', weights_plot='static/weights_plot_final.png')

@app.route('/simulate_result', methods=['POST'])
def simulate_result():
    try:
        global trained_weights
        if not trained_weights:
            return render_template('Simulate.html', 
                                message="Error: No hay pesos entrenados disponibles",
                                dm_plot='static/dm_plot_final.png', 
                                weights_plot='static/weights_plot_final.png')

        num_neurons = int(network_config['num_neurons'])
        
        # Convertir el vector de entrada
        input_vector = np.array([float(x) for x in request.form['vector'].split(',')])
        weights = np.array(trained_weights)
        
        print("Vector de entrada:", input_vector)
        print("Forma de los pesos:", weights.shape)
        print("Forma del vector de entrada:", input_vector.shape)

        # Calcular distancias con manejo de nan
        distances = []
        for i in range(num_neurons):
            dist = euclidean_distance(input_vector, weights[:, i])
            if np.isnan(dist):  # Si la distancia es nan, asignar un valor muy grande
                dist = float('inf')
            distances.append(dist)
            print(f"Distancia a neurona {i}: {dist}")

        # Encontrar la neurona ganadora ignorando nan
        valid_distances = [d for d in distances if not np.isnan(d)]
        if not valid_distances:
            return render_template('Simulate.html', 
                                message="Error: No se pudieron calcular distancias válidas",
                                dm_plot='static/dm_plot_final.png', 
                                weights_plot='static/weights_plot_final.png')

        winner_idx = np.nanargmin(distances)  # usar nanargmin en lugar de argmin
        winning_neuron = f"Neurona vencedora: {winner_idx} (distancia: {distances[winner_idx]:.4f})"
        
        print("Neurona ganadora:", winner_idx)
        print("Distancias:", distances)

        return render_template('Simulate.html', 
                            message=winning_neuron,
                            dm_plot='static/dm_plot_final.png', 
                            weights_plot='static/weights_plot_final.png')

    except Exception as e:
        print("Error en simulate_result:", str(e))
        return render_template('Simulate.html', 
                            message=f"Error: {str(e)}",
                            dm_plot='static/dm_plot_final.png', 
                            weights_plot='static/weights_plot_final.png')

def euclidean_distance(a, b):
    try:
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        # Verificar si hay valores nan en los vectores
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return float('inf')
        return np.sqrt(np.sum((a - b) ** 2))
    except Exception as e:
        print("Error en euclidean_distance:", str(e))
        raise

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    

if __name__ == '__main__':
    app.run(debug=True)
