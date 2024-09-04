# TP_Final_INF_420

Aluno: Gabriel Moreira Marques

Este projeto é uma implementação de classificação de imagens usando os modelos AlexNet e ResNet-18 no dataset Places365. O projeto inclui scripts para rodar a classificação, calcular a acurácia e gerar gráficos comparativos.

Você pode baixar o dataset aqui: https://drive.google.com/drive/folders/117keBBxt5nBZ3LdV1IE5qBbdNbuOqnVh?usp=drive_link


## Requisitos

1. Python 3.6 ou superior
2. Instalar as dependências listadas no arquivo `requirements.txt`

## Instalação

1. Clone o repositório:
    ```sh
    git clone https://github.com/eugabrielm/TP_Final_INF_420.git
    cd seu-repositorio
    ```

2. Crie um ambiente virtual (opcional, mas recomendado):
    ```sh
    python -m venv venv
    source venv/bin/activate  # No Windows use `venv\Scripts\activate`
    ```

3. Instale as dependências:
    ```sh
    pip install -r requirements.txt
    ```

## Execução

### 1. Rodar o código de classificação

OBS: Esse código gerará um documento com as classificações mais o resultado do grad-CAM para cada uma das 5 melhores classificação de cada imagem no dataset, o resultado final ocupará muito espaço, por isso coloquei a tabela de classificação das duas redes no repositorio.

Execute o script de classificação com os parâmetros desejados:
```sh
python places365_classification_dataset.py --input_dir /mnt/d/TP_Final_INF_420/dataset --output_dir /mnt/d/TP_Final_INF_420/results_top5 --arch alexnet
```


### 2. Calcular a acurácia

Execute o script de cálculo de acurácia com os caminhos para os arquivos CSV:
```sh
python accuracy.py --baseline_csv diretorio/para/baseline_csv --resnet18_csv diretorio/para/results_resnet18_csv --alexnet_csv diretorio/para/results_alexnet_csv
```