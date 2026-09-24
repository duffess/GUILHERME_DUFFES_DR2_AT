"""YOLOv4-tiny vs SSD MobileNetV2; tracking IoU com trilhas e avaliação por GT."""
import argparse
from collections import deque
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from comum import DADOS, MODELOS, RESULTADOS, abrir_video, escritor, salvar, json_salvar, csv_salvar

def iou(a,b):
    x,y = max(a[0],b[0]), max(a[1],b[1])
    x2,y2 = min(a[0]+a[2],b[0]+b[2]), min(a[1]+a[3],b[1]+b[3])
    inter = max(0,x2-x)*max(0,y2-y)
    return inter / max(1, a[2]*a[3]+b[2]*b[3]-inter)

class Detector:
    def __init__(self, nome):
        self.nome = nome
        self.classes = (MODELOS / "coco.names").read_text().splitlines()
        if nome == "yolo":
            self.arquivo = MODELOS / "yolov4-tiny.weights"
            self.net = cv2.dnn.readNetFromDarknet(np.fromfile(MODELOS / "yolov4-tiny.cfg", np.uint8), np.fromfile(self.arquivo, np.uint8))
        else:
            self.arquivo = MODELOS / "ssd_mobilenet_v2.pb"
            self.net = cv2.dnn.readNetFromTensorflow(np.fromfile(self.arquivo, np.uint8), np.fromfile(MODELOS / "ssd_mobilenet_v2.pbtxt", np.uint8))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # Conta pesos treináveis: em BatchNorm, gamma/beta são parâmetros,
        # enquanto média/variância são buffers fixos do modelo congelado.
        self.parametros = 0
        for nome in self.net.getLayerNames():
            camada = self.net.getLayer(self.net.getLayerId(nome))
            tamanhos = [int(b.size) for b in camada.blobs]
            if camada.type == "BatchNorm" and len(tamanhos) == 4 and len(set(tamanhos)) == 1:
                self.parametros += tamanhos[0] * 2
            else:
                self.parametros += sum(tamanhos)

    def detectar(self, frame):
        h,w = frame.shape[:2]
        caixas, scores, classes = [], [], []
        if self.nome == "yolo":
            self.net.setInput(cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True))
            saidas = self.net.forward(self.net.getUnconnectedOutLayersNames())
            for saida in saidas:
                for det in saida:
                    classe = int(np.argmax(det[5:]))
                    # Region do OpenCV já fornece det[5:] = objeto * classe.
                    score = float(det[5+classe])
                    if score >= .5:
                        cx,cy,bw,bh = det[:4] * np.array([w,h,w,h])
                        caixas.append([int(cx-bw/2),int(cy-bh/2),int(bw),int(bh)])
                        scores.append(score)
                        classes.append(classe)
        else:
            # Este grafo frozen do TensorFlow tem pré-processamento interno e
            # espera pixels RGB em [0,255], como image_tensor do TF Object Detection API.
            self.net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (300,300), swapRB=True))
            # IDs oficiais COCO têm lacunas; converte para a lista contígua de 80 classes.
            ids = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
            for det in self.net.forward().reshape(-1,7):
                if det[2] >= .5 and int(det[1]) in ids:
                    x,y,x2,y2 = det[3:7]*np.array([w,h,w,h])
                    caixas.append([int(x),int(y),int(x2-x),int(y2-y)])
                    scores.append(float(det[2]))
                    classes.append(ids.index(int(det[1])))
        # NMS por classe, limiar IoU 0.4 em ambos os modelos.
        resultado = []
        for classe in sorted(set(classes)):
            indices = [i for i,c in enumerate(classes) if c == classe]
            manter = cv2.dnn.NMSBoxes([caixas[i] for i in indices], [scores[i] for i in indices], .5, .4)
            for j in np.asarray(manter).reshape(-1):
                i = indices[j]
                resultado.append((caixas[i], scores[i], classe))
        return resultado

class Rastreador:
    def __init__(self, max_ausente=10):
        self.tracks = {}
        self.proximo = 1
        self.entradas = self.saidas = 0
        self.max_ausente = max_ausente

    def atualizar(self, deteccoes, frame):
        pares = sorted([(iou(t['caixa'],d[0]), tid, j)
                        for tid,t in self.tracks.items() for j,d in enumerate(deteccoes)
                        if t['classe'] == d[2]], reverse=True)
        usados_t, usados_d = set(), set()
        for score, tid, j in pares:
            if score < .3 or tid in usados_t or j in usados_d:
                continue
            self.tracks[tid].update(caixa=deteccoes[j][0], ultimo=frame)
            usados_t.add(tid)
            usados_d.add(j)
        for j,(caixa,score,classe) in enumerate(deteccoes):
            if j not in usados_d:
                self.tracks[self.proximo] = dict(caixa=caixa, classe=classe, ultimo=frame, trilha=deque())
                self.proximo += 1
                self.entradas += 1
        for tid in list(self.tracks):
            t = self.tracks[tid]
            if frame-t['ultimo'] > self.max_ausente:
                del self.tracks[tid]
                self.saidas += 1
                continue
            if t['ultimo'] == frame:
                x,y,w,h = t['caixa']
                t['trilha'].append((frame, (x+w//2,y+h//2)))
            while t['trilha'] and frame-t['trilha'][0][0] >= 30:
                t['trilha'].popleft()
        return {tid:t for tid,t in self.tracks.items() if t['ultimo'] == frame}

def main(args):
    cv2.setNumThreads(4)
    resumos = []
    # GT opcional: {"0": [{"id": 1, "bbox": [x,y,w,h]}], ...}, apenas pessoas.
    gt = json.loads(args.gt.read_text()) if args.gt else None
    for nome in ("yolo", "ssd"):
        detector = Detector(nome)
        rastreador = Rastreador()
        cap = abrir_video(args.fonte)
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = None
        tempos, tempos_e2e, logs, id_gt, switches, avaliados, frames_gt = [], [], [], {}, 0, 0, 0
        tp = fp = fn = 0
        previsoes = []
        try:
            for n in range(args.frames):
                ok, frame = cap.read()
                if not ok:
                    break
                inicio_e2e = perf_counter()
                if n == 0:
                    print(f"{nome}: shape={frame.shape}, dtype={frame.dtype}")
                    detector.detectar(frame)  # Aquecimento
                    writer = escritor(RESULTADOS / "ex3" / f"{nome}.avi", fps_video, frame.shape[1::-1])
                inicio = perf_counter()
                deteccoes = detector.detectar(frame)
                ms = (perf_counter()-inicio)*1000
                tempos.append(ms)
                if gt is not None and str(n) in gt:
                    from scipy.optimize import linear_sum_assignment
                    reais = gt[str(n)]
                    pessoas = [d for d in deteccoes if d[2] == 0]
                    acertos = 0
                    if reais and pessoas:
                        custos = np.array([[1-iou(g['bbox'],d[0]) for d in pessoas] for g in reais])
                        custos[custos > .5] = 1000000
                        a,b = linear_sum_assignment(custos)
                        acertos = int(sum(custos[i,j] <= .5 for i,j in zip(a,b)))
                    tp += acertos; fp += len(pessoas)-acertos; fn += len(reais)-acertos
                for (x,y,w,h),score,classe in deteccoes:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(frame,f"{detector.classes[classe]} {score:.2f}",(x,max(15,y-4)),0,.45,(0,255,0),1)
                if nome == "yolo":
                    # Contagem de pedestres; demais classes permanecem no feed de detecção.
                    tracks = rastreador.atualizar([d for d in deteccoes if d[2] == 0], n)
                    previsoes.append({"frame":n,"tracks":[{"id":tid,"bbox":t['caixa']} for tid,t in tracks.items()]})
                    for tid,t in tracks.items():
                        cor = (int(tid*53%256),int(tid*97%256),int(tid*151%256))
                        pontos = np.int32([p for _,p in t['trilha']])
                        if len(pontos)>1:
                            cv2.polylines(frame,[pontos],False,cor,2)
                        x,y,_,_ = t['caixa']
                        cv2.putText(frame,f"ID {tid}",(x,y+20),0,.6,cor,2)
                    if gt is not None and str(n) in gt:
                        frames_gt += 1
                        from scipy.optimize import linear_sum_assignment
                        gt_atual = gt[str(n)]
                        ids_tracks = list(tracks)
                        if gt_atual and ids_tracks:
                            custos = np.array([[1-iou(g['bbox'],tracks[tid]['caixa']) for tid in ids_tracks] for g in gt_atual])
                            custos[custos > .5] = 1000000
                            linhas_gt, col_tracks = linear_sum_assignment(custos)
                            associados = [(gt_atual[i]['id'],ids_tracks[j]) for i,j in zip(linhas_gt,col_tracks) if custos[i,j] <= .5]
                        else:
                            associados = []
                        for gid,tid in associados:
                            if gid in id_gt and id_gt[gid] != tid:
                                switches += 1
                            id_gt[gid] = tid
                            avaliados += 1
                    texto = f"Entradas {rastreador.entradas} | Saidas {rastreador.saidas}"
                    cv2.putText(frame,texto,(10,30),0,.65,(0,255,255),2)
                    print(f"Frame {n}: {len(deteccoes)} deteccoes | {texto} | {ms:.2f} ms")
                    logs.append({"frame":n,"ms":ms,"deteccoes":len(deteccoes),"entradas":rastreador.entradas,"saidas":rastreador.saidas})
                else:
                    print(f"SSD frame {n}: {len(deteccoes)} deteccoes | {ms:.2f} ms")
                    logs.append({"frame":n,"ms":ms,"deteccoes":len(deteccoes)})
                writer.write(frame)
                tempos_e2e.append((perf_counter()-inicio_e2e)*1000)
                if n == min(30,args.frames-1):
                    salvar(RESULTADOS / "ex3" / f"{nome}.jpg",frame)
                if not args.sem_janelas:
                    cv2.imshow(nome,frame)
                    if cv2.waitKey(1)&255 == ord('q'):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        if not tempos:
            raise ValueError("Vídeo sem frames.")
        resumo = {"modelo":nome,"frames":len(tempos),"latencia_ms":float(np.mean(tempos)),
                  "fps":1000/float(np.mean(tempos)),"parametros_treinaveis":int(detector.parametros),
                  "latencia_pipeline_ms":float(np.mean(tempos_e2e)),
                  "fps_pipeline":1000/float(np.mean(tempos_e2e)),
                  "arquivo_mb":detector.arquivo.stat().st_size/1024**2,
                  "fonte":str(args.fonte),"id_switches":switches if gt and frames_gt else None,
                  "id_switches_min":switches/(frames_gt/fps_video/60) if gt and frames_gt else None,
                  "associacoes_gt":avaliados,"frames_anotados":frames_gt}
        resumo.update(precisao_pessoa=tp/max(1,tp+fp) if gt else None,recall_pessoa=tp/max(1,tp+fn) if gt else None,tp=tp,fp=fp,fn=fn)
        resumos.append(resumo)
        csv_salvar(RESULTADOS / "ex3" / f"{nome}_frames.csv",logs)
        if nome == 'yolo':
            json_salvar(RESULTADOS / 'ex3' / 'tracks.json',previsoes)
    json_salvar(RESULTADOS / "ex3" / "comparacao.json",resumos)
    print("Modelo | FPS inferência | ms inferência | FPS pipeline | parâmetros treináveis | MB em disco | precision pessoa | recall pessoa")
    for r in resumos:
        precisao = f"{r['precisao_pessoa']:.3f}" if r['precisao_pessoa'] is not None else "N/D"
        recall = f"{r['recall_pessoa']:.3f}" if r['recall_pessoa'] is not None else "N/D"
        print(f"{r['modelo']:6} | {r['fps']:.2f} | {r['latencia_ms']:.2f} | {r['fps_pipeline']:.2f} | {r['parametros_treinaveis']} | {r['arquivo_mb']:.2f} | {precisao} | {recall}")
    print("ID switches/min:",resumos[0]['id_switches_min'] if gt else "N/D - requer identidades anotadas, não inferível de IDs criados")
    mais_rapido = min(resumos,key=lambda r:r['latencia_ms'])['modelo']
    conclusao = f"{mais_rapido} foi o mais rápido neste CPU/vídeo. Candidato para menor latência; escolha embarcada exige medir recall e energia no dispositivo."
    print(conclusao)
    (RESULTADOS / "ex3" / "conclusao.txt").write_text(conclusao,encoding="utf-8")
    # Entradas = novos tracks; saídas = timeout, não cruzamento de uma fronteira.
    # Oclusões podem fragmentar tracks e inflar contagens. Um ID novo NÃO prova
    # ID switch: a métrica exige correspondência com identidade de referência.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonte",default=str(DADOS / "pedestres_publico.avi"))
    parser.add_argument("--frames",type=int,default=100)
    parser.add_argument("--gt",type=Path)
    parser.add_argument("--sem-janelas",action="store_true")
    main(parser.parse_args())
