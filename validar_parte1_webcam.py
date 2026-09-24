"""Valida a rubrica 1 ao vivo sem gravar frames da webcam no disco."""
import argparse
import json
from time import perf_counter
import cv2
import numpy as np
from comum import RESULTADOS, json_salvar

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--mostrar", action="store_true")
    args = p.parse_args()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    tempos, forma, dtype, deteccoes, desvios, comandos = [], None, None, 0, [], []
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a webcam {args.camera}.")
        print(f"OpenCV {cv2.__version__} | câmera {args.camera} aberta")
        for i in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                print(f"Leitura encerrada no frame {i}.")
                break
            if forma is None:
                forma, dtype = list(frame.shape), str(frame.dtype)
                print(f"Frame 0: shape={tuple(forma)} dtype={dtype}")
            inicio = perf_counter()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mascara = cv2.inRange(hsv, np.array([100,150,50]), np.array([140,255,255]))
            mascara = cv2.erode(mascara, None, iterations=2)
            mascara = cv2.dilate(mascara, None, iterations=2)
            contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            validos = [c for c in contornos if cv2.contourArea(c) > 500]
            linha = frame.copy()
            if validos:
                c = max(validos, key=cv2.contourArea)
                m = cv2.moments(c)
                if m['m00'] > 0:
                    cx, cy = int(m['m10']/m['m00']), int(m['m01']/m['m00'])
                    erro = (cx-frame.shape[1]/2)/(frame.shape[1]/2)
                    omega = float(np.clip(-0.6*erro, -0.6, 0.6))
                    desvios.append(float(erro)); comandos.append(omega); deteccoes += 1
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(linha,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.circle(linha,(cx,cy),6,(0,0,255),-1)
                    cv2.arrowedLine(linha,(frame.shape[1]//2,frame.shape[0]//2),(cx,frame.shape[0]//2),(0,255,255),2)
                    cv2.putText(linha,f"erro_x={erro:.2f} omega={omega:.2f}",(10,25),0,.6,(0,255,255),2)
            tempos.append((perf_counter()-inicio)*1000)
            if args.mostrar:
                cv2.imshow("Webcam HSV | azul + comando de direcao",linha)
                cv2.imshow("Mascara HSV",mascara)
                if cv2.waitKey(1)&255 == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if not tempos:
        raise RuntimeError("A webcam não forneceu frames.")
    resumo = {"frames_lidos":len(tempos),"shape":forma,"dtype":dtype,
              "frames_com_alvo":deteccoes,"latencia_media_segmentacao_ms":float(np.mean(tempos)),
              "erro_x_medio":float(np.mean(desvios)) if desvios else None,
              "omega_medio":float(np.mean(comandos)) if comandos else None,
              "frames_salvos":False}
    print(json.dumps(resumo,ensure_ascii=False,indent=2))
    json_salvar(RESULTADOS/"rubricas_parte1_webcam.json",resumo)

if __name__ == "__main__":
    main()
