# Visão Computacional para Robótica

Repositório dos exercícios de retorno da disciplina. O projeto reúne métodos clássicos e redes neurais para calibração, classificação, detecção, rastreamento e segmentação de imagens, usando OpenCV e TensorFlow/Keras.

## Conteúdo

| Exercício | Tema | Principais resultados |
|---|---|---|
| 1 | Calibração de câmera e realidade aumentada | Erro de reprojeção, correção `undistort` e cubo 3D estimado com `solvePnP` e projetado com `projectPoints`. Detalhes em [EX1.md](EX1.md); imagens e métricas em `resultados/ex1/`. |
| 2 | Classificação com OpenCV DNN e pipeline | Comparação MobileNetV2 em OpenCV DNN e Keras, além do notebook de integração. Documentação em [EX2.md](EX2.md); previsões e saídas em `resultados/ex2/`. |
| 3 | Detecção e rastreamento | Comparação YOLOv4-tiny/SSD MobileNetV2 e associação de objetos por IoU. Documentação em [EX3.md](EX3.md); vídeos, tabelas e trilhas em `resultados/ex3/`. |
| 4 | Segmentação semântica | Comparação DeepLabV3 e limiarização HSV em cinco cenas. Documentação em [EX4.md](EX4.md); painéis, áreas por classe e tempos em `resultados/ex4/`. |

O [relatório integrativo](RELATORIO_INTEGRATIVO.md) reúne o diagrama, as métricas, a análise de viabilidade embarcada, a arquitetura proposta e as lacunas previstas para a DR4. [VALIDACAO_RUBRICAS.md](VALIDACAO_RUBRICAS.md) relaciona os critérios às evidências disponíveis. As fontes e licenças dos dados estão em [DATASETS_PUBLICOS.md](DATASETS_PUBLICOS.md).

## Organização

- `ex1_calibracao.py`, `ex2_classificacao.py`, `ex3_deteccao.py` e `ex4_segmentacao.py`: scripts principais.
- `preparar_*.py` e `comum.py`: download, extração e preparação dos dados e modelos.
- `dados/`: manifestos e, após preparação local, datasets de entrada.
- `modelos/`: configurações, pesos e arquivos exportados dos modelos.
- `resultados/ex1/` a `resultados/ex4/`: saídas separadas por exercício.
- `resultados/tps/` e `resultados/cnn/`: medições e experimentos complementares dos TPs.
- `Exercicio_2_pipeline.ipynb`: sequência interativa do pipeline integrado do Exercício 2.

## Ambiente e preparação

Os datasets completos, pesos de redes, vídeos e o ambiente virtual não fazem parte do repositório para reduzir seu tamanho. Os scripts de preparação baixam os arquivos públicos necessários. No Windows, a partir desta pasta, use Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe preparar_dados.py publicos
.\.venv\Scripts\python.exe preparar_dados.py detectores
.\.venv\Scripts\python.exe preparar_calibracao_publica.py
.\.venv\Scripts\python.exe preparar_mot.py
.\.venv\Scripts\python.exe preparar_cenas_ex4.py
.\.venv\Scripts\python.exe ex2_classificacao.py exportar
```

## Execução

Os comandos abaixo reproduzem os experimentos sem abrir janelas interativas:

```powershell
.\.venv\Scripts\python.exe ex1_calibracao.py calibrar --pasta dados\tabuleiro_publico --padrao 9 6 --quadrado 1 --unidade quadrados --sem-janelas
.\.venv\Scripts\python.exe ex1_calibracao.py ar --fonte dados\tabuleiro_publico.avi --inicio 160 --frames 100 --sem-janelas
.\.venv\Scripts\python.exe ex2_classificacao.py comparar
.\.venv\Scripts\python.exe ex3_deteccao.py --fonte dados\mot_pedestres.avi --frames 179 --gt dados\mot_gt.json --sem-janelas
.\.venv\Scripts\python.exe ex4_segmentacao.py --sem-janelas
```

O notebook pode ser executado no mesmo ambiente. Os testes usam bases públicas e, em alguns experimentos, dados sintéticos. A calibração e a realidade aumentada foram avaliadas em vídeo público gravado; os parâmetros dessa calibração não correspondem à imagem Penn-Fudan usada no pipeline integrado. Assim, os resultados não representam uma validação completa com a webcam nem uma medição em hardware embarcado.
