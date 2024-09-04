import pandas as pd
import matplotlib.pyplot as plt
import argparse


# Definir os argumentos de linha de comando
parser = argparse.ArgumentParser(description='Processar parâmetros de entrada para arquivos CSV.')
parser.add_argument('--baseline_csv', type=str, required=True, help='Caminho para o arquivo baseline.csv')
parser.add_argument('--resnet18_csv', type=str, required=True, help='Caminho para o arquivo results_resnet18_top5.csv')
parser.add_argument('--alexnet_csv', type=str, required=True, help='Caminho para o arquivo results_alexnet_top5.csv')
args = parser.parse_args()

# Carregamento dos dados
baseline = pd.read_csv(args.baseline_csv)
resnet18_results = pd.read_csv(args.resnet18_csv)
alexnet_results = pd.read_csv(args.alexnet_csv)
# Converter as colunas para lowercase para evitar erros de formatação
baseline.columns = [col.lower() for col in baseline.columns]
resnet18_results.columns = [col.lower() for col in resnet18_results.columns]
alexnet_results.columns = [col.lower() for col in alexnet_results.columns]

# Função para calcular acurácia top k
def calculate_top_k_accuracy(results, baseline, k):
    merged = results.merge(baseline, left_on='image name', right_on='image_name', how='inner')
    grouped = merged.groupby('image name')
    
    correct_predictions = 0
    total_images = len(grouped)

    for image_name, group in grouped:
        true_label = group['label'].iloc[0]  
        top_k_predictions = group.nsmallest(k, 'rank')['predicted label'].tolist()  

        if true_label in top_k_predictions:
            correct_predictions += 1

    accuracy = correct_predictions / total_images
    return accuracy



# Cálculo das acurácias para AlexNet e ResNet18
top_k_accuracies_resnet18 = [calculate_top_k_accuracy(resnet18_results, baseline, k) for k in [1, 3, 5]]
top_k_accuracies_alexnet = [calculate_top_k_accuracy(alexnet_results, baseline, k) for k in [1, 3, 5]]

# Ajuste de espaçamento entre os gráficos
plt.subplots_adjust(hspace=0.5)  # Aumenta o espaço entre as subplots

# Definir a largura das barras e a posição no eixo x
bar_width = 0.35
x = [1, 3, 5]
x_positions = [i - bar_width/2 for i in x]

# Gráfico de acurácia
plt.figure(figsize=(10, 6))
plt.bar(x_positions, top_k_accuracies_resnet18, width=bar_width, label='ResNet18')
plt.bar([p + bar_width for p in x_positions], top_k_accuracies_alexnet, width=bar_width, label='AlexNet')
plt.title('Comparação de Acurácia')
plt.xlabel('k')
plt.ylabel('Acurácia')
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.show()

