# AT de Visão Computacional para Robótica

Este diretório reúne os quatro exercícios, os scripts usados, os dados públicos e os resultados que podem ser conferidos. Para começar a apresentação, siga o **roteiro curto** abaixo; para detalhes técnicos, abra o relatório do exercício correspondente.

## Roteiro de apresentação

### 1. Calibração e realidade aumentada

Mostre [EX1.md](EX1.md), depois `resultados/ex1/original_corrigida.jpg` e `resultados/ex1/cubo.jpg`. Explique que a calibração obteve RMS global de 0,953 px em 16 vistas e que o cubo foi projetado em vídeo com `solvePnP` e `projectPoints`. **Ressalva:** os dados vieram de uma sequência pública; não são calibração da webcam.

### 2. OpenCV DNN e pipeline

Mostre [EX2.md](EX2.md), as imagens anotadas em `resultados/ex2/opencv/` e `resultados/ex2/pipeline_final.jpg`. Compare a tabela OpenCV DNN/Keras e percorra o notebook [Exercicio_2_pipeline.ipynb](Exercicio_2_pipeline.ipynb). **Ressalva:** o pipeline roda e mede as etapas, mas usa parâmetros de calibração de outra câmera; portanto, não alegue correção geométrica validada nessa imagem.

### 3. YOLO, SSD e rastreamento

Mostre [EX3.md](EX3.md), `resultados/ex3/yolo.jpg`, `resultados/ex3/ssd.jpg` e, se houver tempo, os vídeos `yolo.avi` e `ssd.avi`. Use `comparacao.json`/`terminal.txt` para FPS, latência e métricas; `tracks.json` e `yolo.avi` mostram IDs, trilhas e contagem. **Ressalva:** a taxa de ID switches por minuto é extrapolada de um vídeo curto.

### 4. Segmentação semântica

Mostre [EX4.md](EX4.md), um painel `resultados/ex4/*_painel.jpg` e `resultados/ex4/areas.csv`. Compare DeepLabV3 com HSV e explique que são cinco cenas públicas. **Ressalva:** as classes VOC usadas não incluem pista/calçada e não há ground truth pareado para mIoU.

### 5. Fechamento

Use o [Relatório integrativo](RELATORIO_INTEGRATIVO.md) para conectar resultados, custo embarcado, proposta de arquitetura e limitações. Consulte [Validação das rubricas](VALIDACAO_RUBRICAS.md) se o professor quiser verificar cada critério. O [catálogo de datasets](DATASETS_PUBLICOS.md) registra origem e uso dos dados.

## Mapa de pastas

| Caminho | Conteúdo |
|---|---|
| `ex1_calibracao.py` … `ex4_segmentacao.py` | Scripts principais dos exercícios |
| `preparar_*.py`, `comum.py` | Preparação de dados e funções auxiliares |
| `dados/` | Datasets e vídeos de entrada, com proveniência quando aplicável |
| `modelos/` | Pesos e configurações usados pelos scripts |
| `resultados/ex1/` … `resultados/ex4/` | Saídas finais, tabelas, logs e vídeos por exercício |
| `resultados/tps/`, `resultados/cnn/` | Evidências complementares dos TPs e do experimento CNN |
| `requirements.txt` | Dependências Python do projeto |

## Reproduzir as demonstrações

O Git contém código, documentação, métricas e evidências compactas. Datasets completos, vídeos e pesos ficam fora do commit para manter o clone leve; os manifests de origem permanecem versionados. No PowerShell, a partir desta pasta, prepare o ambiente e recupere os ativos quando precisar reproduzir as execuções:

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

Depois, execute os experimentos:

```powershell
.\.venv\Scripts\python.exe ex1_calibracao.py calibrar --pasta dados\tabuleiro_publico --padrao 9 6 --quadrado 1 --unidade quadrados --sem-janelas
.\.venv\Scripts\python.exe ex1_calibracao.py ar --fonte dados\tabuleiro_publico.avi --inicio 160 --frames 100 --sem-janelas
.\.venv\Scripts\python.exe ex2_classificacao.py comparar
.\.venv\Scripts\python.exe ex3_deteccao.py --fonte dados\mot_pedestres.avi --frames 179 --gt dados\mot_gt.json --sem-janelas
.\.venv\Scripts\python.exe ex4_segmentacao.py --sem-janelas
```

O notebook do Exercício 2 pode ser aberto no Jupyter com o mesmo ambiente. As demonstrações usam datasets públicos locais e alguns experimentos complementares usam dados sintéticos; não descreva esses resultados como coleta própria. A calibração e o AR são vídeos gravados, e o conjunto de resultados não é uma validação integral em tempo real na webcam.

## O que abrir primeiro

Se houver pouco tempo, esta é a sequência mínima de evidências:

1. `resultados/ex1/original_corrigida.jpg` e `resultados/ex1/cubo.jpg`.
2. `resultados/ex2/opencv/` e `resultados/ex2/pipeline_final.jpg`.
3. `resultados/ex3/yolo.jpg`, `resultados/ex3/ssd.jpg` e `resultados/ex3/comparacao.json`.
4. `resultados/ex4/mot15_tud_0044_painel.jpg` e `resultados/ex4/areas.csv`.
5. `RELATORIO_INTEGRATIVO.md` e `VALIDACAO_RUBRICAS.md`.
