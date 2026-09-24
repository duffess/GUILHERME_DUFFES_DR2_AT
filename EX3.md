# Exercício 3 — Detecção YOLO/SSD e rastreamento

## O que foi implementado

[`ex3_deteccao.py`](ex3_deteccao.py) executa YOLOv4-tiny e SSD MobileNetV2 com OpenCV DNN, no CPU, sobre o mesmo vídeo público MOT15. Cada detector desenha caixas, rótulos e confiança, aplica NMS por classe com IoU 0,4 e registra latência por frame. O vídeo de saída contém as anotações. O código mede também o pipeline com desenho, rastreamento e gravação, além da inferência isolada.

No YOLO, o rastreador associa detecções da classe pessoa entre frames por IoU guloso (limiar 0,3), mantém uma trilha de até 30 frames e exibe IDs, entradas e saídas. Um ID novo conta como entrada; uma faixa que fica ausente por mais de 10 frames conta como saída. Isso aproxima surgimento/desaparecimento no campo de visão, mas não detecta cruzamento de uma linha de entrada/saída. Oclusões podem criar fragmentos e supercontagem.

## Dados e execução

Foi usado o trecho TUD-Stadtmitte do MOT15, com vídeo de 179 frames, 25 FPS e anotações de identidade. As caixas de referência estão em `dados/mot_gt.json`; a origem e a sequência estão descritas em `dados/mot/ORIGEM.json` e `dados/mot/seqinfo.ini`. É um conjunto público substituindo a webcam, conforme combinado para este trabalho; portanto, estes resultados caracterizam esse vídeo e este computador, não uma coleta do ambiente do aluno.

Para repetir a execução completa no Windows:

```powershell
.\.venv\Scripts\python.exe ex3_deteccao.py --fonte dados\mot_pedestres.avi --frames 179 --gt dados\mot_gt.json --sem-janelas
```

Remova `--sem-janelas` para abrir as janelas durante a execução. Os pesos e configurações ficam em `modelos/`. A medição começa após uma inferência de aquecimento. FPS de inferência é 1000 dividido pela latência média; FPS do pipeline inclui anotação, rastreamento quando aplicável e gravação, mas não inclui leitura/decodificação do vídeo.

## Resultados medidos

| Modelo | Inferência média | FPS inferência | FPS pipeline | Parâmetros treináveis estimados | Arquivo de pesos | Precisão pessoa | Recall pessoa |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv4-tiny | 64,50 ms | 15,50 | 13,81 | 6.056.618 | 23,13 MB | 0,987 | 0,798 |
| SSD MobileNetV2 | 62,48 ms | 16,00 | 14,88 | 16.837.567 | 66,46 MB | 0,989 | 0,687 |

A precisão e o recall usam correspondência gulosa ótima por Hungarian com IoU mínimo de 0,5 entre caixas detectadas e caixas anotadas, considerando a classe pessoa. YOLO obteve 923 TP, 12 FP e 233 FN; SSD obteve 794 TP, 9 FP e 362 FN. Os parâmetros são contados a partir dos blobs carregados no OpenCV: pesos e biases são contados integralmente; em BatchNorm, gamma e beta entram como parâmetros e média/variância são tratados como buffers fixos. O tamanho em disco é o arquivo de pesos, sem a configuração/labels.

Neste teste, SSD foi cerca de 2 ms mais rápido por inferência e teve precisão semelhante. YOLO, porém, teve recall maior em 11 pontos percentuais e o modelo de pesos ocupou aproximadamente um terço do espaço do SSD, com cerca de 36% dos parâmetros. Para um alvo embarcado com memória limitada ou maior prioridade para encontrar pedestres, YOLO é a escolha mais adequada entre estes dois resultados. Se cada milissegundo for prioritário e o recall medido for suficiente, SSD também é uma opção. A diferença de velocidade é pequena e pode variar com hardware, backend, threads e condições de execução; nenhum consumo de energia ou hardware de 5 W foi medido. Ambos ficaram perto de 15 FPS, então o ensaio não demonstra 30 FPS em tempo real.

## Rastreamento e validação

O rastreador YOLO registrou 17 novas trilhas/entradas e 11 expirações/saídas até o frame 178. Comparado à identidade de referência com IoU 0,5, foram observados 11 ID switches em 179 frames anotados (7,16 s a 25 FPS), ou 92,18 switches/minuto por extrapolação. Essa taxa por minuto é muito instável em um clipe tão curto; deve ser lida como resultado descritivo do trecho, não como estimativa de desempenho geral. Um ID novo criado pelo rastreador não é por si só um ID switch: a métrica compara a associação com a identidade anotada.

Saídas de validação:

- `resultados/ex3/yolo.avi`: detecções YOLO com IDs, trilhas e contagem cumulativa.
- `resultados/ex3/ssd.avi`: detecções SSD com caixas, classe e confiança.
- `resultados/ex3/yolo.jpg` e `resultados/ex3/ssd.jpg`: quadros anotados para inspeção.
- `resultados/ex3/yolo_frames.csv` e `resultados/ex3/ssd_frames.csv`: latência e quantidade de detecções por frame; o CSV YOLO inclui entradas e saídas.
- `resultados/ex3/comparacao.json`: métricas agregadas, inclusive avaliação contra ground truth.
- `resultados/ex3/tracks.json`: IDs e caixas YOLO por frame.
- `resultados/ex3/terminal.txt`: tempos por frame e tabela final impressa.

## Considerações éticas para contagem por drone

O sistema pode estimar fluxos agregados de pedestres para planejamento urbano, desde que a finalidade seja clara e haja autorização para operar o drone e processar imagens. A contagem deve evitar identificação individual: processar no próprio dispositivo, não guardar vídeo bruto nem rostos, reter somente estatísticas agregadas pelo menor tempo necessário e sinalizar a coleta conforme as regras locais. O desempenho pode variar por iluminação, altura, ângulo, oclusão e grupos sub-representados; por isso, contagens não devem embasar decisões individuais ou policiamento automatizado. É necessário avaliar erros em diferentes locais e horários, manter revisão humana para decisões relevantes e publicar limites e finalidade do sistema.
