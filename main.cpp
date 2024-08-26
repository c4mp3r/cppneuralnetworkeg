#include <QCoreApplication>
#include <QFile>
#include <QTextStream>
#include <QStringList>
#include <QtDebug>
#include <cmath>
#include <vector>
#include <QRandomGenerator>

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Neural Network structure
class NeuralNetwork {
public:
    NeuralNetwork();
    void initialize();
    void feedforward();
    void readCsvAndFeedforward(const QString &filename);

private:
    static const int INPUT_NEURONS = 5;
    static const int HIDDEN_NEURONS = 3;
    static const int OUTPUT_NEURONS = 5;

    std::vector<double> input;
    std::vector<double> hidden;
    std::vector<double> output;
    std::vector<std::vector<double>> weight_input_hidden;
    std::vector<std::vector<double>> weight_hidden_output;
    std::vector<double> bias_hidden;
    std::vector<double> bias_output;
};

NeuralNetwork::NeuralNetwork() {
    input.resize(INPUT_NEURONS);
    hidden.resize(HIDDEN_NEURONS);
    output.resize(OUTPUT_NEURONS);
    weight_input_hidden.resize(INPUT_NEURONS, std::vector<double>(HIDDEN_NEURONS));
    weight_hidden_output.resize(HIDDEN_NEURONS, std::vector<double>(OUTPUT_NEURONS));
    bias_hidden.resize(HIDDEN_NEURONS);
    bias_output.resize(OUTPUT_NEURONS);
    initialize();
}

void NeuralNetwork::initialize() {
    QRandomGenerator *generator = QRandomGenerator::global();

    for (int i = 0; i < INPUT_NEURONS; i++) {
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            weight_input_hidden[i][j] = generator->generateDouble() - 0.5;
        }
    }
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        for (int j = 0; j < OUTPUT_NEURONS; j++) {
            weight_hidden_output[i][j] = generator->generateDouble() - 0.5;
        }
    }
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        bias_hidden[i] = generator->generateDouble() - 0.5;
    }
    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        bias_output[i] = generator->generateDouble() - 0.5;
    }
}

void NeuralNetwork::feedforward() {
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        hidden[i] = 0;
        for (int j = 0; j < INPUT_NEURONS; j++) {
            hidden[i] += input[j] * weight_input_hidden[j][i];
        }
        hidden[i] += bias_hidden[i];
        hidden[i] = sigmoid(hidden[i]);
    }

    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            output[i] += hidden[j] * weight_hidden_output[j][i];
        }
        output[i] += bias_output[i];
        output[i] = sigmoid(output[i]);
    }
}

void NeuralNetwork::readCsvAndFeedforward(const QString &filename) {
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCritical() << "Error: Could not open file" << filename;
        return;
    }

    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList tokens = line.split(',');

        if (tokens.size() != INPUT_NEURONS) {
            qCritical() << "Error: Expected" << INPUT_NEURONS << "inputs per line.";
            continue;
        }

        for (int i = 0; i < INPUT_NEURONS; i++) {
            input[i] = tokens[i].toDouble();
        }

        feedforward();

        qDebug() << "Output:";
        for (int i = 0; i < OUTPUT_NEURONS; i++) {
            qDebug() << output[i];
        }
    }

    file.close();
}

int main(int argc, char *argv[]) {
    QCoreApplication a(argc, argv);

    NeuralNetwork nn;

    QString filename = "input_data.csv";
    nn.readCsvAndFeedforward(filename);

    return a.exec();
}
