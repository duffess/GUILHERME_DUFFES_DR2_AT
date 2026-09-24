"""Extrai vistas reais variadas de uma mesma câmera pública, sem duplicá-las."""
import cv2
import numpy as np
from comum import DADOS, baixar, abrir_video, salvar, json_salvar

def main():
    fonte = baixar("https://raw.githubusercontent.com/smidm/video2calibration/master/example_input/chessboard.avi", DADOS / "tabuleiro_publico.avi")
    pasta = DADOS / "tabuleiro_publico"
    pasta.mkdir(exist_ok=True)
    cap = abrir_video(fonte)
    selecionadas = []
    try:
        for n in range(160, 2040, 60):
            cap.set(cv2.CAP_PROP_POS_FRAMES,n)
            ok, frame = cap.read()
            if not ok:
                continue
            achou, pts = cv2.findChessboardCorners(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(9,6))
            if achou:
                salvar(pasta / f"frame_{n:04}.png",frame)
                selecionadas.append(n)
    finally:
        cap.release()
    json_salvar(pasta / "ORIGEM.json", {"url":"https://github.com/smidm/video2calibration", "tipo":"Vídeo REAL público; não é webcam do aluno", "frames":selecionadas, "padrao":[9,6], "escala":"Tamanho físico não informado; translação em unidades de quadrado"})
    print(f"{len(selecionadas)} vistas reais extraídas.")
    if len(selecionadas)<15:
        raise RuntimeError("Menos de 15 vistas válidas.")

if __name__ == "__main__":
    main()
