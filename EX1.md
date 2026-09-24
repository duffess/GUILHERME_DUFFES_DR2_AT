# Exercício 1 — Calibração de câmera e realidade aumentada

Este exercício estima os parâmetros intrínsecos e a distorção de uma câmera, corrige imagens com `cv2.undistort` e usa a pose do tabuleiro para sobrepor um cubo virtual. O fluxo foi executado em um vídeo público do projeto `video2calibration`, que contém um tabuleiro filmado pela mesma câmera em várias posições. Esse teste demonstra o método, mas não calibra a webcam do aluno.

## Arquivos

- [`ex1_calibracao.py`](ex1_calibracao.py) contém os três modos. `capturar` mostra a webcam e permite salvar uma imagem quando `S` é pressionado e os 42 cantos internos de um tabuleiro 8×7 são detectados. `calibrar` aceita no mínimo 15 imagens, refina os cantos com `cornerSubPix`, estima a matriz K e a distorção com `calibrateCamera`, calcula erro por imagem e corrige uma imagem. `ar` reutiliza a calibração, estima pose com `solvePnP`, projeta um cubo de uma casa com `projectPoints` e grava a pose de cada frame.
- [`preparar_calibracao_publica.py`](preparar_calibracao_publica.py) baixa o vídeo público do mesmo projeto e extrai quadros espaçados nos quais o padrão 9×6 é detectado. O tamanho físico dos quadrados não está publicado, então as translações são expressas em unidades de quadrado. Esta sequência não representa a webcam.
- [`comum.py`](comum.py) contém funções compartilhadas de caminhos, leitura e gravação de imagens, downloads verificáveis via `curl` e gravação de vídeos, também usadas pelos exercícios seguintes.
- [`requirements.txt`](requirements.txt) lista as dependências Python do diretório.
- `dados/tabuleiro_publico.avi` é o vídeo-fonte baixado. `dados/tabuleiro_publico/` contém 16 imagens espaçadas e `ORIGEM.json` registra sua proveniência e padrão detectado.
- `resultados/ex1/calibracao.npz` contém K, coeficientes de distorção, resolução, dimensões do padrão e unidade usada, para que a estimativa de pose possa reutilizá-los.
- `resultados/ex1/metricas.json` guarda parâmetros e medidas da calibração; `reprojecao.csv` traz o erro de cada vista em pixels.
- `resultados/ex1/original_corrigida.jpg` compara lado a lado a imagem de entrada e a versão corrigida com `cv2.undistort`.
- `resultados/ex1/cubo.avi` é a sequência anotada da realidade aumentada; `cubo.jpg` é um frame com o cubo sobreposto.
- `resultados/ex1/pose_tempos.csv` registra o estado do tabuleiro, tempo por frame e os seis valores estimados (`rvec` e `tvec`) por frame. `pose_resumo.json` resume cobertura e tempos; `pose_terminal.txt` guarda a impressão de rotação e translação no terminal.

## O que os parâmetros representam

A matriz intrínseca é

```text
[ fx   0  cx ]
[  0  fy  cy ]
[  0   0   1 ]
```

`fx` e `fy` são as distâncias focais medidas em pixels nos eixos horizontal e vertical; `cx` e `cy` indicam o ponto principal da imagem. Os cinco coeficientes retornados aqui seguem a ordem do OpenCV `k1, k2, p1, p2, k3`: `k1`, `k2` e `k3` descrevem distorção radial da lente; `p1` e `p2` descrevem distorção tangencial, associada ao desalinhamento entre lente e sensor. Uma regra prática inicial é erro médio de reprojeção abaixo de 1 pixel; valores menores que 0,5 pixel são desejáveis em algumas aplicações de precisão, mas nenhum limite isolado certifica segurança robótica. Também é necessário validar em imagens e posições que não entraram no ajuste.

`solvePnP` fornece `rvec`, a orientação 3D em forma de vetor de Rodrigues, e `tvec`, a posição do padrão relativa à câmera. A calibração usa coordenadas no tabuleiro; a escala e, portanto, a unidade de `tvec` só têm significado físico se a aresta real de cada quadrado for medida e informada. As imagens públicas usadas aqui não publicam essa medida. Para medir em metros, capture um tabuleiro conhecido e execute `calibrar --quadrado 0.025 --unidade m` se cada casa tiver exatamente 2,5 cm. No código, as faces visíveis do cubo usam cores diferentes para ajudar a distinguir as arestas.

## Como executar

Ative o ambiente virtual com Python 3.11 e instale as dependências definidas em `requirements.txt`:

```powershell
cd ATRETORNO
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Extraia as imagens públicas, calibre e processe o mesmo vídeo para a sobreposição. A opção sem janelas salva os resultados no disco e facilita a execução em lote:

```powershell
.venv\Scripts\python.exe preparar_calibracao_publica.py
.venv\Scripts\python.exe ex1_calibracao.py calibrar --pasta dados/tabuleiro_publico --padrao 9 6 --quadrado 1 --unidade quadrados --sem-janelas
.venv\Scripts\python.exe ex1_calibracao.py ar --fonte dados/tabuleiro_publico.avi --inicio 160 --frames 100 --sem-janelas
```

Para fazer a calibração solicitada com a webcam, imprima um tabuleiro com 8 colunas e 7 linhas de quadrados, fixe foco/zoom, conecte-o à webcam e grave vistas variadas por toda a imagem. A câmera não deve mudar de resolução entre a calibração e o vídeo da pose:

```powershell
.venv\Scripts\python.exe ex1_calibracao.py capturar --fonte 0
.venv\Scripts\python.exe ex1_calibracao.py calibrar --pasta dados/tabuleiro --quadrado 0.025 --unidade m
.venv\Scripts\python.exe ex1_calibracao.py ar --fonte 0
```

Pressione `S` para cada captura válida e `Q` para sair. Mova o tabuleiro para perto/longe, gire-o e cubra centro, bordas e cantos. Para usar outro vídeo, substitua `--fonte 0` pelo caminho do arquivo. Preserve os quadros de calibração e anote a medida verdadeira do quadrado.

## Execução e limites observados

Na sequência pública foram aceitas 16 de 16 imagens espaçadas. O ajuste retornou RMS global de **0,953 px** e média de **0,862 px** nos RMS individuais por imagem, em resolução de 1280×720. Esses valores são compatíveis com uma demonstração clássica e ficam abaixo da referência prática de 1 px, mas descrevem o ajuste nessa câmera pública, não a webcam do aluno. O ajuste levou cerca de **1,02 s** neste computador. O RMS global e a média dos valores por imagem usam agregações diferentes e, portanto, não precisam ser iguais.

O vídeo do tabuleiro foi processado em 100 quadros a partir do frame 160; os cantos apareceram em **97 quadros**. A rotação e translação são registradas por quadro e a sobreposição foi salva. O algoritmo reporta ausência de pose nos frames sem detecção, em vez de reutilizar uma posição antiga. Neste lote, a detecção e o refinamento dos cantos levaram em média **27,82 ms** por quadro (mediana **25,38 ms**); é medição desta máquina e desta resolução, sem exibição de janelas.

O tracker só mantém a pose enquanto o padrão está visível. A calibração pode falhar com reflexo, desfoque, padrões parcialmente cortados, mudança de foco ou resolução e poses pouco variadas. O conjunto público serve como validação reprodutível do código. Para apresentar calibração da webcam, use os três comandos de captura acima e guarde a saída e as novas evidências.

## Fontes e transparência

- Código, vídeo e padrão público de calibração: [repositório video2calibration](https://github.com/smidm/video2calibration). A documentação do projeto especifica o padrão 9×6 e seu procedimento de gravação. Vídeo público usado como dado demonstrativo; nenhuma imagem foi atribuída à webcam do aluno.
- API e convenções da calibração: [tutorial oficial de calibração de câmera do OpenCV](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) e documentação oficial de [`solvePnP` / `projectPoints`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
- Código e texto foram preparados com auxílio de IA generativa (OpenAI Codex, 22–23/09/2026); as métricas numéricas acima foram produzidas pela execução local descrita neste arquivo. A autoria, conferência e adequação da entrega são responsabilidade do aluno.
