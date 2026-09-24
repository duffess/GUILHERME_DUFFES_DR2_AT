# Exercício 4 — Segmentação semântica

## Item A

[`ex4_segmentacao.py`](ex4_segmentacao.py) executa DeepLabV3 pré-treinada via torchvision CPU e compara sua máscara com uma regra HSV para tons de verde nas mesmas imagens. O modelo produz classes VOC coloridas, overlay semitransparente e porcentagem de área por classe. `preparar_cenas_ex4.py` baixa três amostras KITTI urbanas e extrai dois quadros externos do vídeo MOT15 já presente nos dados. As origens estão registradas em [`dados/cenas_externas/ORIGEM.json`](dados/cenas_externas/ORIGEM.json); a página de atribuição identifica as três cenas KITTI como imagens urbanas de benchmark, com licença CC BY-NC-SA 3.0 para uso acadêmico. [Atribuição das amostras KITTI](https://csundergrad.science.uoit.ca/courses/csci3240u/latest/labs/data/kitti-samples/ATTRIBUTION.html)

Para refazer a execução no Windows, a partir desta pasta:

```powershell
.\.venv\Scripts\python.exe preparar_cenas_ex4.py
.\.venv\Scripts\python.exe ex4_segmentacao.py --sem-janelas
```

O modelo baixado fica em `modelos/deeplabv3.pth`. Os cinco painéis lado a lado ficam em `resultados/ex4/*_painel.jpg`; `*_classes.png` guarda IDs de classe, `*_cores.png` guarda a máscara colorida, `legenda.json` relaciona classes a cores, `areas.csv` registra percentuais e `tempos.csv` registra latência por imagem. `terminal.txt` contém as porcentagens impressas.

Nas cinco imagens, DeepLabV3 levou em média 497,98 ms por imagem em CPU; a máscara HSV levou 0,80 ms. Na amostra KITTI `um_000032.png`, a rede atribuiu 1,38% a ônibus, 1,46% a carro, 0,05% a pessoa e 97,11% a background. Os pesos VOC usados não têm classes para pista e calçada, e o background não deve ser interpretado como superfície dirigível. O HSV destaca pixels verdes, não reconhece a categoria vegetação nem distingue pista de calçada. Como não há máscaras semânticas pareadas para estas cinco imagens, não foi calculada acurácia por pixel ou mIoU.

## Item B

O relatório obrigatório de mais de 800 palavras, com diagrama do pipeline, tabela de métricas, análise de orçamento de 5 W, arquitetura urbana proposta e três lacunas para DR4 está em [`RELATORIO_INTEGRATIVO.md`](RELATORIO_INTEGRATIVO.md). Ele distingue medições executadas de propostas e registra os limites das amostras sintéticas e públicas.
