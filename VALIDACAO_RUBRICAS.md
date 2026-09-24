# Validação das rubricas — Parte 1

Auditoria dos quatro blocos e de cada pergunta da rubrica fornecida. O status avalia as evidências existentes nesta pasta, nos TPs originais e nas execuções locais; presença de código sem uma execução verificável não foi tratada automaticamente como demonstração.

**Legenda:** **D** = demonstrado por código e evidência; **P** = parcial, com a limitação descrita; **ND** = não demonstrado.

## Rubrica 1 — OpenCV e visão clássica

| Status | Critério e evidência |
|---|---|
| **D** | Ambiente Python/OpenCV, stream de câmera, limpeza de recursos e inspeção de frame. Os originais `GUILHERME_DUFFES_DR2_TP1/ex1.py` e `ex4.py` usam webcam e liberam `VideoCapture`/janelas. Executei a webcam por 60 frames com `ATRETORNO/validar_parte1_webcam.py`; terminal e JSON registram `shape=(480,640,3)`, `dtype=uint8`, 60 frames, OpenCV 4.11 e recursos encerrados. Nenhum frame foi salvo. Evidências: `resultados/rubricas_parte1_webcam_terminal.txt` e `resultados/rubricas_parte1_webcam.json`. |
| **D** | Conversão HSV/LAB e manipulação de canais. `GUILHERME_DUFFES_DR2_TP1/ex2.py` separa e altera canais. A medição comparativa em `resultados/tps/metricas.json` registra HSV 1,33 ms, LAB 2,42 ms e erros médios de reconstrução de 0,287 e 0,364; também registra o efeito da alteração do canal. |
| **D** | Limiar global, adaptativo e Otsu, escolha e geometria de contornos. `ex3.py` executa os três limiares, filtra área e extrai perímetro, razão e centroide. `medir_tps.py` registra tempos, foreground e contornos; o código explica em que condição cada limiar é apropriado. As métricas de foreground não são acurácia, pois não há máscara de referência. |
| **P** | Captura, HSV e segmentação em rastreamento ao vivo com saída de controle. `GUILHERME_DUFFES_DR2_TP1/ex4.py` calcula centroide, erro horizontal normalizado, direção e seta; `validar_parte1_webcam.py` inclui também comando proporcional `omega`. O teste ao vivo de 60 frames não encontrou um alvo azul, então não observamos um comando calculado sobre objeto real; nenhum motor foi ligado. O código e a lógica estão presentes, mas falta uma execução com alvo colorido visível para demonstrar o caminho completo. |

## Rubrica 2 — CNN, treinamento e representações

| Status | Critério e evidência |
|---|---|
| **D** | CNN Keras treinada e avaliada, curvas de treino/validação e regularização com BatchNorm/Dropout. `complemento_cnn.py` treina três variantes, salva `resultados/cnn/curvas.png` e métricas. No conjunto sintético de setas, acurácia de teste foi 91,88% (baseline), 56,25% (regularizada) e 100% (augmentation). A execução demonstra o processo; o dado sintético não representa cenas robóticas. |
| **D** | CNN como extratora, PCA e comparação com ORB. `resultados/cnn/metricas.json` registra classificador linear sobre as features: acurácia 100% para CNN e 61,88% para ORB; silhouette após PCA 0,711 e -0,006. `resultados/cnn/pca_cnn_orb.png` contém a visualização. |
| **D** | Data augmentation com rotação, zoom/escala e flip com inversão de rótulo. `aumentar()` em `complemento_cnn.py` inverte esquerda/direita quando espelha. A acurácia subiu 43,75 pontos percentuais frente à variante regularizada e 8,13 pontos frente ao baseline, nesse teste sintético; o gap treino-validação final foi 0,4625 na variante regularizada e 0,0000 na augmentation. |
| **D** | CNN treinada do zero como extratora, PCA e comparação de separabilidade com descritor clássico. A CNN e o ORB são comparados sobre as mesmas classes sintéticas e divisões fixas, com PCA, silhouette e classificador linear em `resultados/cnn/metricas.json`. |

As curvas mostram treino perto de 100% enquanto a validação cai na variante regularizada (final 53,75%, com loss de validação subindo de 0,588 para 0,623), evidência de generalização ruim neste conjunto. A baseline valida entre 85,63% e 95,63%; a variante com augmentation termina treino/validação em 100%, com loss de validação 0,171 e teste 100%. Como treino usa amplitudes menores de rotação/escala que validação/teste, o diagnóstico é específico ao experimento sintético e não deve ser extrapolado para imagens reais.

## Rubrica 3 — detecção, rastreamento e reconhecimento

| Status | Critério e evidência |
|---|---|
| **D** | Detecção integrada com IDs persistentes e taxa de ID switches. `ex3_deteccao.py` associa pessoas por IoU; `resultados/ex3/tracks.json` guarda IDs e `comparacao.json` mede 11 switches em 7,16 s (92,18/min extrapolados). O vídeo curto torna a taxa/min muito instável; isso está documentado em `EX3.md`. |
| **D** | SVM próprio com HOG, janela deslizante, métricas e análise de custo frente ao YOLO. `AT/AT-EX1/item_b.py` usa 100 positivos/100 negativos, testa 40 recortes e registra acurácia, precisão e recall de 1,000; a janela avaliou 1.076 posições e reteve 20 caixas após NMS. O código compara o custo por janela/vetores de suporte com uma passagem do YOLO. **Limite:** a janela não foi cronometrada diretamente contra YOLO e o teste de 40 recortes é pequeno. |
| **D** | Filtro de Kalman para suavização/predição e papel de Q, R e P. `AT/AT-EX2/rastreamento.py` documenta as matrizes. No vídeo sintético de 180 frames, RMSE do centro foi 2,45 px para CamShift, 1,56 px para previsão e 1,29 px após correção; trajetória de referência é conhecida pela equação geradora. |
| **D** | Janela deslizante HOG+SVM e comparação de custo computacional com YOLO. Implementação em `AT/AT-EX1/item_b.py`, contagem de janelas e comentário de complexidade. A comparação é analítica, não um benchmark de latência pareado. |
| **P** | Reconhecimento facial com biblioteca de alto nível, latência por frame e ética. `GUILHERME_DUFFES_DR2_TP2/ex2b.py` usa `face_recognition` e mede/imprime a latência por frame. A discussão ética específica está em `RELATORIO_INTEGRATIVO.md`; falta registrar uma execução aferida, então o requisito combinado permanece parcial. |

## Rubrica 4 — calibração, DNN, detectores e integração

| Status | Critério e evidência |
|---|---|
| **D** | Calibração com tabuleiro, erro de reprojeção e undistort. `ex1_calibracao.py` usou 16 vistas do mesmo vídeo público: RMS global 0,953 px e média RMS por vista 0,862 px. `reprojecao.csv` e `original_corrigida.jpg` documentam os resultados. A calibração é daquela câmera pública, não da webcam. |
| **D** | `solvePnP` e objeto 3D sobreposto com pose. O vídeo `resultados/ex1/cubo.avi` tem o cubo projetado; pose foi detectada em 97/100 frames. `pose_terminal.txt` e `pose_tempos.csv` guardam rvec/tvec. É vídeo gravado, não uma execução validada ao vivo na webcam. |
| **D** | Modelo pré-treinado OpenCV DNN comparado com outro backend. MobileNetV2 com mesmos pesos: OpenCV DNN 14,74 ms, RSS 94,1 MB e top-1 80%; Keras 18,93 ms, RSS 398,2 MB e top-1 80% em dez imagens. Evidência em `EX2.md` e `resultados/ex2/opencv.json`/`keras.json`; teste pequeno. |
| **P** | Pipeline de técnicas dos TPs com latência por etapa. O notebook do Ex. 2 executa undistort, HSV, ORB, HOG+SVM e DNN em 97,86 ms. Porém aplica parâmetros de calibração de outra câmera ao frame Penn-Fudan; a chamada roda, mas sua validade geométrica não foi demonstrada. Limite descrito em `EX2.md`. |
| **D** | YOLO e SSD, comparação de desempenho e justificativa embarcada. 179 frames do MOT15 produziram latências de 64,50/62,48 ms, precisão 0,987/0,989, recall 0,798/0,687 e pesos 23,13/66,46 MB. Vídeos, CSVs e anotação ground truth estão em `resultados/ex3/`. São imagens públicas reais, não capturadas no ambiente do aluno; consumo em watts não foi medido. |
| **D** | Segmentação semântica pré-treinada comparada a HSV. `ex4_segmentacao.py` processa cinco cenas externas, salva overlay e percentuais por classe, e compara com limiar HSV na mesma imagem. Evidências em `resultados/ex4/` e `EX4.md`. Pesos VOC não têm pista/calçada e não há ground truth pareado para mIoU. |
| **D** | Relatório técnico integrativo com diagrama, métricas, análise embarcada, arquitetura e lacunas para DR4. `RELATORIO_INTEGRATIVO.md` tem 2.000+ palavras, os cinco tópicos obrigatórios e métricas coletadas de vários TPs. A análise de 5 W é baseada em latência/tamanho; consumo elétrico real não foi aferido. |
| **D** | Organização, documentação e evidências auditáveis. Scripts, relatórios, dados com origem, resultados em CSV/JSON e imagens/vídeos estão separados por exercício. Scripts do Ex. 4 e da validação webcam passaram em `py_compile`; os vídeos/saídas principais foram conferidos. |
| **D** | Erro de reprojeção por imagem e correção radial. `reprojecao.csv` traz as 16 vistas e `cv2.undistort` produz o comparativo. A evidência usa a câmera pública do tabuleiro; não é calibração da webcam. |
| **P** | Pipeline completo medindo latência de cada etapa. O notebook mede cada etapa e o total de 97,86 ms, mas mantém a incompatibilidade entre câmera calibrada e imagem avaliada. A execução sequencial funciona; a correção geométrica não tem validade metrológica nesse par. |
| **D** | AR usando `solvePnP` e `projectPoints`. Cubo de uma casa, faces coloridas e vetores estimados por frame estão implementados em `ex1_calibracao.py`; vídeo e logs são reprodutíveis no material público. |
| **D** | DeepLabV3, máscara colorida, overlay e comparação com HSV. Cinco imagens estão anotadas; os limites de classes VOC e ausência de mIoU estão informados no relatório. |

## Pendências antes de marcar todos os itens como demonstrados

1. Repetir a segmentação ao vivo do TP1 com um objeto azul visível e guardar somente métricas/terminal, sem armazenar a imagem da webcam. Isso fecha o caminho de centroide até comando de direção; não requer ligar um motor.
2. Registrar uma execução do reconhecimento facial com latência por frame. O cronômetro já existe no código e a análise ética foi acrescentada ao relatório; o requisito permanece parcial até haver evidência de execução.
3. Se a avaliação exigir validade geométrica no pipeline do Exercício 2, calibrar a webcam e executar o notebook com frame da mesma câmera; hoje esse é o principal limite metrológico.
