"""Executa scripts reais dos TPs sem bloquear em janelas; mede novamente técnicas."""
import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import cv2
import numpy as np
from comum import ROOT, RESULTADOS, ler, salvar, medir, json_salvar, ambiente

def main():
    ambiente()
    cv2.setNumThreads(4)
    workspace = ROOT.parent
    out = RESULTADOS / "tps"
    out.mkdir(exist_ok=True)
    # Mantém os scripts originais; substitui somente a exibição por arquivos.
    show, wait, destroy = cv2.imshow, cv2.waitKey, cv2.destroyAllWindows
    cwd = Path.cwd()
    try:
        cv2.waitKey = lambda *_: ord('q')
        cv2.destroyAllWindows = lambda: None
        for pasta,script in [('GUILHERME_DUFFES_DR2_TP1','ex2.py'),('GUILHERME_DUFFES_DR2_TP1','ex3.py'),
                              ('GUILHERME_DUFFES_DR2_TP2','ex1.py'),('GUILHERME_DUFFES_DR2_TP2','ex3.py')]:
            destino = out / pasta / script.replace('.py','')
            destino.mkdir(parents=True,exist_ok=True)
            contador = [0]
            def exibir(nome,img):
                contador[0] += 1
                salvar(destino / f"painel_{contador[0]}.jpg",img)
            cv2.imshow = exibir
            os.chdir(workspace / pasta)
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                runpy.run_path(script,run_name='__main__')
            (destino / 'terminal.txt').write_text(log.getvalue(),encoding='utf-8')
            print(pasta,script,': executado')
    finally:
        os.chdir(cwd)
        cv2.imshow,cv2.waitKey,cv2.destroyAllWindows = show,wait,destroy
    img = ler(workspace / 'GUILHERME_DUFFES_DR2_TP1' / 'euela.jpg')
    # Registra resolução: tempos de entradas diferentes não são ranking universal.
    dados = []
    for nome,conversao,inversa in [('HSV',cv2.COLOR_BGR2HSV,cv2.COLOR_HSV2BGR),('LAB',cv2.COLOR_BGR2LAB,cv2.COLOR_LAB2BGR)]:
        convertido,ms = medir(lambda:cv2.cvtColor(img,conversao))
        volta = cv2.cvtColor(convertido,inversa)
        erro = float(np.abs(img.astype(float)-volta.astype(float)).mean())
        canais = cv2.split(convertido)
        alterados = list(canais)
        indice = 1 if nome=='HSV' else 0
        alterados[indice] = np.clip(canais[indice].astype(float)*.5,0,255).astype(np.uint8)
        alterada = cv2.cvtColor(cv2.merge(alterados),inversa)
        salvar(out / f'{nome}_canais.jpg',np.hstack(canais))
        salvar(out / f'{nome}_alterado.jpg',alterada)
        dados.append(dict(tecnica=nome,ms=ms,qualidade=f'MAE ida/volta={erro:.4f}; MAE canal alterado={np.abs(img.astype(float)-alterada).mean():.4f}',shape=list(img.shape)))
    cinza = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    funcoes = {'Global':lambda:cv2.threshold(cinza,127,255,cv2.THRESH_BINARY)[1],
               'Adaptativo':lambda:cv2.adaptiveThreshold(cinza,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2),
               'Otsu':lambda:cv2.threshold(cinza,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]}
    for nome,funcao in funcoes.items():
        mascara,ms = medir(funcao)
        contornos,_ = cv2.findContours(mascara,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        geometrias = []
        for c in contornos:
            area = cv2.contourArea(c)
            if area>200:
                x,y,w,h = cv2.boundingRect(c); m = cv2.moments(c)
                geometrias.append(dict(area=area,perimetro=cv2.arcLength(c,True),razao=w/h,centro=[m['m10']/m['m00'],m['m01']/m['m00']]))
        json_salvar(out / f'{nome}_geometria.json',geometrias)
        dados.append(dict(tecnica=nome,ms=ms,qualidade=f'foreground={np.mean(mascara>0):.3%}; contornos>200px={len(geometrias)}; acurácia N/D sem máscara GT',shape=list(img.shape)))
    img = ler(workspace / 'GUILHERME_DUFFES_DR2_TP2' / 'cena1.jpg',cv2.IMREAD_GRAYSCALE)
    for nome,det in [('ORB',cv2.ORB_create()),('SIFT',cv2.SIFT_create()),('AKAZE',cv2.AKAZE_create())]:
        (kp,desc),ms = medir(lambda:det.detectAndCompute(img,None))
        dados.append(dict(tecnica=nome,ms=ms,qualidade=f'{len(kp)} pontos; {desc.nbytes if desc is not None else 0} bytes de descritores; acurácia N/D',shape=list(img.shape)))
    json_salvar(out / 'metricas.json',dados)
    for d in dados:
        print(d)
    # Global: luz homogênea e limiar conhecido. Otsu: histograma bimodal.
    # Adaptativo: variação local de iluminação, com maior custo e ruído/textura.

if __name__=='__main__':
    main()
