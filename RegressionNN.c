#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

// Структура для представления нейрона
typedef struct Neuron {
    double *weights;
    double bias;
    double output;
    double delta;
} Neuron;

// Структура для представления слоя
typedef struct Layer {
    int num_neurons;
    Neuron *neurons;
} Layer;

// Структура для представления модели
typedef struct Model {
    int num_layers;
    Layer *layers;
} Model;

// Аргументы для передачи в поток
typedef struct ThreadArgs {
    Model *model;
    double **data;
    int start_index;
    int end_index;
    int epochs;
    double learning_rate;
} ThreadArgs;

// Функция для создания новой модели
Model *create_model(int num_layers, int *layer_sizes) {
    Model *model = (Model *)malloc(sizeof(Model));
    model->num_layers = num_layers;
    model->layers = (Layer *)malloc(num_layers * sizeof(Layer));
    
    for (int i = 0; i < num_layers; i++) {
        Layer *layer = &model->layers[i];
        layer->num_neurons = layer_sizes[i];
        layer->neurons = (Neuron *)malloc(layer_sizes[i] * sizeof(Neuron));
        
        for (int j = 0; j < layer_sizes[i]; j++) {
            Neuron *neuron = &layer->neurons[j];
            neuron->weights = (double *)malloc((i ? layer_sizes[i - 1] : 1) * sizeof(double));
            neuron->bias = 0.0;
            neuron->output = 0.0;
            neuron->delta = 0.0;
            
            for (int k = 0; k < (i ? layer_sizes[i - 1] : 1); k++) {
                neuron->weights[k] = (rand() / (double)RAND_MAX) * 2.0 - 1.0; // Random initialization
            }
        }
    }
    
    return model;
}

// Функция для освобождения памяти, занятой моделью
void destroy_model(Model *model) {
    for (int i = 0; i < model->num_layers; i++) {
        Layer *layer = &model->layers[i];
        for (int j = 0; j < layer->num_neurons; j++) {
            Neuron *neuron = &layer->neurons[j];
            free(neuron->weights);
        }
        free(layer->neurons);
    }
    free(model->layers);
    free(model);
}

// Функция для прямого распространения
void forward_propagation(Model *model, double *input) {
    for (int i = 0; i < model->num_layers; i++) {
        Layer *layer = &model->layers[i];
        
        for (int j = 0; j < layer->num_neurons; j++) {
            Neuron *neuron = &layer->neurons[j];
            double sum = neuron->bias;
            
            for (int k = 0; k < (i ? model->layers[i - 1].num_neurons : 1); k++) {
                sum += (i ? model->layers[i - 1].neurons[k].output : input[k]) * neuron->weights[k];
            }
            
            neuron->output = tanh(sum); // Activation function
        }
    }
}

// Функция для обратного распространения ошибки
void backpropagation(Model *model, double *expected_output, double learning_rate) {
    // Calculate deltas for the output layer
    Layer *output_layer = &model->layers[model->num_layers - 1];
    for (int i = 0; i < output_layer->num_neurons; i++) {
        Neuron *neuron = &output_layer->neurons[i];
        double delta = (expected_output[i] - neuron->output) * (1 - pow(neuron->output, 2)); // Derivative of activation function
        neuron->bias -= learning_rate * delta;
        
        for (int j = 0; j < model->layers[model->num_layers - 2].num_neurons; j++) {
            Neuron *prev_neuron = &model->layers[model->num_layers - 2].neurons[j];
            neuron->weights[j] -= learning_rate * delta * prev_neuron->output;
        }
    }
    
    // Calculate deltas for hidden layers
    for (int l = model->num_layers - 2; l > 0; l--) {
        Layer *layer = &model->layers[l];
        Layer *next_layer = &model->layers[l + 1];
        
        for (int i = 0; i < layer->num_neurons; i++) {
            Neuron *neuron = &layer->neurons[i];
            double delta_sum = 0.0;
            
            for (int j = 0; j < next_layer->num_neurons; j++) {
                Neuron *next_neuron = &next_layer->neurons[j];
                delta_sum += next_neuron->weights[i] * next_neuron->delta;
            }
            
            double delta = delta_sum * (1 - pow(neuron->output, 2)); // Derivative of activation function
            neuron->bias -= learning_rate * delta;
            
            for (int j = 0; j < model->layers[l - 1].num_neurons; j++) {
                Neuron *prev_neuron = &model->layers[l - 1].neurons[j];
                neuron->weights[j] -= learning_rate * delta * prev_neuron->output;
            }
        }
    }
}

// Функция для обучения модели в потоке
void *train_thread(void *args) {
    ThreadArgs *thread_args = (ThreadArgs *)args;
    Model *model = thread_args->model;
    double **data = thread_args->data;
    int start_index = thread_args->start_index;
    int end_index = thread_args->end_index;
    int epochs = thread_args->epochs;
    double learning_rate = thread_args->learning_rate;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int sample = start_index; sample < end_index; sample++) {
            forward_propagation(model, data[sample]);
            backpropagation(model, data[sample + 1], learning_rate);
        }
    }
    
    return NULL;
}

// Функция для параллельного обучения модели
void parallel_train(Model *model, double **training_data, int num_samples, int epochs, double learning_rate, int num_threads) {
    pthread_t threads[num_threads];
    ThreadArgs args[num_threads];
    
    int samples_per_thread = num_samples / num_threads;
    int remainder = num_samples % num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        args[i].model = model;
        args[i].data = training_data;
        args[i].start_index = i * samples_per_thread + (i < remainder ? i : remainder);
        args[i].end_index = (i + 1) * samples_per_thread + (i < remainder ? i + 1 : remainder);
        args[i].epochs = epochs;
        args[i].learning_rate = learning_rate;
        
        pthread_create(&threads[i], NULL, train_thread, &args[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

// Функция для оценки модели на тестовых данных
double test_model(Model *model, double **test_data, int num_test_samples) {
    double mse = 0.0;
    
    for (int i = 0; i < num_test_samples; i++) {
        forward_propagation(model, test_data[i]);
        double predicted_value = model->layers[model->num_layers - 1].neurons[0].output;
        double actual_value = test_data[i][1];
        mse += pow(predicted_value - actual_value, 2);
    }
    
    return mse / num_test_samples;
}

// Функция для предсказания значений
double predict(Model *model, double *input) {
    forward_propagation(model, input);
    return model->layers[model->num_layers - 1].neurons[0].output;
}

int main() {
    srand(time(NULL)); // Инициализация генератора случайных чисел
    
    // Параметры модели
    int num_layers = 3;
    int layer_sizes[] = {1, 5, 1}; // Входной слой, скрытый слой, выходной слой
    
    // Генерация тренировочных данных
    int num_training_samples = 10000;
    double **training_data = (double **)malloc(num_training_samples * sizeof(double *));
    for (int i = 0; i < num_training_samples; i++) {
        training_data[i] = (double *)malloc(2 * sizeof(double));
        training_data[i][0] = rand() / (double)RAND_MAX; // Случайное значение
        training_data[i][1] = training_data[i][0] * 2;  // Линейная зависимость
    }
    
    // Генерация тестовых данных
    int num_test_samples = 1000;
    double **test_data = (double **)malloc(num_test_samples * sizeof(double *));
    for (int i = 0; i < num_test_samples; i++) {
        test_data[i] = (double *)malloc(2 * sizeof(double));
        test_data[i][0] = rand() / (double)RAND_MAX; // Случайное значение
        test_data[i][1] = test_data[i][0] * 2;       // Линейная зависимость
    }
    
    // Создание модели
    Model *model = create_model(num_layers, layer_sizes);
    
    // Параметры обучения
    int epochs = 100;
    double learning_rate = 0.01;
    int num_threads = 4;
    
    // Обучение модели
    parallel_train(model, training_data, num_training_samples, epochs, learning_rate, num_threads);
    
    // Оценка модели
    double mse = test_model(model, test_data, num_test_samples);
    printf("MSE: %.6f\n", mse);
    
    // Предсказание
    double input = 0.5;
    double prediction = predict(model, &input);
    printf("Prediction for input %.2f: %.2f\n", input, prediction);
    
    // Освобождение памяти
    destroy_model(model);
    for (int i = 0; i < num_training_samples; i++) {
        free(training_data[i]);
    }
    free(training_data);
    for (int i = 0; i < num_test_samples; i++) {
        free(test_data[i]);
    }
    free(test_data);
    
    return 0;
}