#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_SEQUENCE_LENGTH 100

// Структура для хранения состояния ячейки RNN
typedef struct {
    float* hidden_state;
} RNNCellState;

// Функция инициализации состояния ячейки RNNCellState
void init_rnn_cell_state(RNNCellState* state, int num_hidden_units) {
    state->hidden_state = (float*)malloc(num_hidden_units * sizeof(float));
    for (int i = 0; i < num_hidden_units; ++i) {
        state->hidden_state[i] = 0.0f;
    }
}

// Освобождение памяти, выделенной под состояние ячейки RNN
void free_rnn_cell_state(RNNCellState* state) {
    if (state->hidden_state != NULL) {
        free(state->hidden_state);
        state->hidden_state = NULL;
    }
}

// Функция активации tanh
float tanh_activation(float x) {
    return (exp(2*x) - 1) / (exp(2*x) + 1);
}

// Обновление состояния ячейки RNNCellState
void rnn_cell_forward(float* input, RNNCellState* state, float* weights, int
 num_input_units, int num_hidden_units) {
    // Инициализация временного вектора для промежуточных расчётов
    float* temp = (float*)malloc((num_input_units + num_hidden_units) * sizeof(float));
    
    // Копирование входного сигнала и текущего скрытого состояния во временный вектора
    for (int i = 0; i < num_input_units; ++i) {
        temp[i] = input[i];
    }
    
    for (int i = 0; i < num_hidden_units; ++i) {
        temp[num_input_units + i] = state->hidden_state[i];
    }
    
    // Вычисление нового скрытого состояния через взвешенную сумму входов и предыдущего скрытого состояния
    for (int i = 0; i < num_hidden_units; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < num_input_units + num_hidden_units; ++j) {
            sum += temp[j] * weights[(i * (num_input_units + num_hidden_units)) + j];
        }
        state->hidden_state[i] = tanh_activation(sum);
    }
    
    free(temp);
}

// Основная функция для выполнения последовательности шагов RNNCellState
void run_rnn_sequence(float** inputs, RNNCellState* state, float* weights, int 
    sequence_length, int num_input_units, int num_hidden_units) {
    for (int t = 0; t < sequence_length; ++t) {
        rnn_cell_forward(inputs[t], state, weights, num_input_units, num_hidden_units);
    }        
}

int main() {
	// Параметры модели
	const int sequence_length = 10;
	const int num_input_units = 5;
	const int num_hidden_units = 3;
	
	// Входные данные
	float** inputs = (float**)malloc(sequence_length * sizeof(float*));
	for (int t = 0; t < sequence_length; ++t) {
	    inputs[t] = (float*)malloc(num_input_units * sizeof(float));
	    for (int i = 0; i < num_input_units; ++i) {
	        inputs[t][i] = rand() % 100 / 100.0f; // Случайное значение от 0 до 1
	    }
	}
	
	// Веса модели
	float* weights = (float*)malloc(num_hidden_units * (num_input_units + num_hidden_units) * sizeof(float));
	for (int i = 0; i < num_hidden_units * (num_input_units + num_hidden_units); ++i) {
	    weights[i] = rand() % 100 / 100.0f; // Случайное значение от 0 до 1
	}
	
	// Состояние ячейки RNN
    RNNCellState state;
    init_rnn_cell_state(&state, num_hidden_units);
    
    // Выполнение последовательности шагов RNN
    run_rnn_sequence(inputs, &state, weights, sequence_length, num_input_units, num_hidden_units);
    
    // Печать финального скрытого состояния
    printf("Final Hidden State:\n");
    for (int i = 0; i < num_hidden_units; ++i) {
        printf("%f ", state.hidden_state[i]);
    }
    printf("\n");
    
    // Освобождение выделенной памяти
    free_rnn_cell_state(&state);
    for (int t = 0; t < sequence_length; ++t) {
        free(inputs[t]);
    }
    free(inputs);
    free(weights);
    
    return 0;
}

