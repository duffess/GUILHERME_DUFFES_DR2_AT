# Exercício 2 — Classificação com DNN e pipeline integrado

## Item A — MobileNetV2, OpenCV DNN e Keras

O classificador usa MobileNetV2 treinada no ImageNet. O mesmo conjunto de pesos Keras foi exportado para ONNX; o grafo ONNX roda no backend CPU do OpenCV DNN. Keras roda nativamente na mesma CPU. Ambos recebem o mesmo blob RGB 224×224 normalizado para [-1, 1], recebem a mesma sequência de dez imagens de validação Imagenette e produzem top-1/top-3. O script mede a execução do `forward` após cinco passagens de aquecimento. A memória é RSS máxima observada pelo processo e inclui o runtime.

| Backend | Latência média por inferência | RSS máximo amostrado | Acurácia top-1 |
|---|---:|---:|---:|
| OpenCV DNN CPU | 14,74 ms | 94,1 MB | 8/10 = 80% |
| Keras CPU | 18,93 ms | 398,2 MB | 8/10 = 80% |

O maior desvio de probabilidade entre os backends foi 0,00000251, consistente com as mesmas previsões. O resultado sugere que, **neste computador, runtime e lote**, OpenCV DNN usou menos RSS e teve menor latência. Embarcados podem favorecer DNN quando o modelo é compatível com OpenCV e o backend/target disponível é mais eficiente; a decisão requer benchmark no hardware final. RSS não é potência, e um conjunto de apenas dez imagens não fornece uma estimativa precisa de acurácia. A listagem completa de previsões, probabilidades e imagens anotadas está em `resultados/ex2/opencv.json`, `keras.json` e nas pastas `opencv/` e `keras/`.

O `cv2.putText` exibe as três classes superiores e confiança em cada imagem. `classes.json` guarda o índice oficial ImageNet usado para descodificar as previsões; `classificacao.json` liga cada imagem ao rótulo esperado. A imagem corretamente classificada entra no cálculo de top-1. O teste usa uma imagem por classe para satisfazer o escopo mínimo do enunciado; substitua por todas as imagens de validação se o professor pedir estimativa robusta.

## Item B — notebook funcional

[`Exercicio_2_pipeline.ipynb`](Exercicio_2_pipeline.ipynb) contém células sequenciais para carregar um frame do Penn-Fudan, corrigir distorção, criar máscara HSV, calcular pontos ORB, detectar pedestres com o HOG e SVM linear padrão do OpenCV, e classificar a ROI principal com a rede DNN do item A. As células exibem as saídas e tempos individuais em ms; a última grava `resultados/ex2/pipeline_final.jpg`, `pipeline_hsv_mask.png` e `pipeline_metricas.json`.

Nesta execução, o frame contém quatro caixas HOG e 500 pontos ORB. A medição com uma passagem de aquecimento excluída foi:

| Etapa | Tempo |
|---|---:|
| Carregar frame do dataset | 5,79 ms |
| `undistort` | 3,73 ms |
| HSV | 1,13 ms |
| ORB | 5,92 ms |
| HOG + SVM | 64,59 ms |
| DNN da ROI | 16,69 ms |
| **Total das etapas** | **97,86 ms** |

## Reprodutibilidade e ressalva de calibração

Ative `.venv` com Python 3.11. Os dados de Imagenette/Penn-Fudan e os pesos do modelo já ficam em `dados/` e `modelos/`; para recomputar as partes A e B:

```powershell
.venv\Scripts\python.exe ex2_classificacao.py exportar
.venv\Scripts\python.exe ex2_classificacao.py comparar
.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name atretorno --display-name "Python 3.11 (ATRETORNO)"
```

Abra `Exercicio_2_pipeline.ipynb`, selecione o kernel `Python 3.11 (ATRETORNO)` e execute todas as células. Se necessário, instale as dependências antes com `python -m pip install -r requirements.txt`. Para refazer o notebook a partir das células documentadas em código, execute `python criar_ex2_notebook.py` e execute todas as células novamente.

O frame Penn-Fudan não tem parâmetros intrínsecos/distortion associados. Os únicos coeficientes do Exercício 1 vieram de outra câmera num vídeo público de tabuleiro. Por isso o notebook executa a chamada exigida `cv2.undistort`, mas identifica explicitamente essa saída como **demonstração funcional sem validade metrológica**. Não se afirma que pixels do Penn-Fudan foram calibrados corretamente. Para resultado geométrico válido, use frame capturado pela webcam e calibre essa mesma webcam com 15 ou mais vistas do tabuleiro.

O classificador ImageNet é genérico: não é um modelo “pedestre versus fundo”. A previsão top-3 da caixa HOG serve para demonstrar o encadeamento, não como rótulo semântico validado para pedestres. Os pesos ImageNet pré-treinados e o formato de exportação ONNX são usados apenas para inferência, não há treino com a imagem avaliada.

## Fontes dos dados/modelo

- [Imagenette, repositório fast.ai](https://github.com/fastai/imagenette): 10 classes de ImageNet, variante 160 pixels, divisão oficial de validação.
- [Penn-Fudan Pedestrian Database](https://www.cis.upenn.edu/~jshi/ped_html/) e exemplo de sua estrutura de imagem/máscara no [tutorial oficial do torchvision](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html).
- [MobileNetV2 original em Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2); exportação para ONNX é necessária para executar os mesmos pesos na API OpenCV DNN.
