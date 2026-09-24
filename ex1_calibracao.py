"""Ex.1 A/B: captura, calibração de câmera e cubo com pose solvePnP."""
import argparse
from time import perf_counter
import cv2
import numpy as np
from comum import DADOS, RESULTADOS, ler, salvar, json_salvar, csv_salvar, abrir_video, escritor

PADRAO = (7, 6)  # Cantos INTERNOS: imprima 8 x 7 quadrados.

def pontos_tabuleiro(quadrado):
    pontos = np.zeros((PADRAO[0]*PADRAO[1], 3), np.float32)
    pontos[:, :2] = np.mgrid[0:PADRAO[0], 0:PADRAO[1]].T.reshape(-1, 2) * quadrado
    return pontos

def cantos(frame):
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    achou, pts = cv2.findChessboardCorners(cinza, PADRAO)
    if achou:
        pts = cv2.cornerSubPix(cinza, pts, (11, 11), (-1, -1),
                              (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    return achou, pts

def capturar(args):
    args.pasta.mkdir(parents=True, exist_ok=True)
    cap = abrir_video(args.fonte)
    print("S salva uma vista válida; Q encerra. Varie inclinação, posição e distância.")
    n = len(list(args.pasta.glob("*.png")))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            achou, pts = cantos(frame)
            anotada = frame.copy()
            if achou:
                cv2.drawChessboardCorners(anotada, PADRAO, pts, achou)
            cv2.putText(anotada, f"Capturas: {n} / minimo 15", (10, 25), 0, 0.6, (0, 255, 0), 2)
            cv2.imshow("Captura - S salvar / Q sair", anotada)
            tecla = cv2.waitKey(1) & 255
            if tecla == ord("s") and achou:
                salvar(args.pasta / f"tabuleiro_{n:03}.png", frame)
                n += 1
                print(f"Captura {n}: shape={frame.shape}, dtype={frame.dtype}")
            if tecla == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def calibrar(args):
    imagens = sorted(p for p in args.pasta.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    objetos, observados, nomes = [], [], []
    tamanho = None
    inicio = perf_counter()
    for path in imagens:
        img = ler(path)
        if tamanho is not None and img.shape[1::-1] != tamanho:
            raise ValueError("Todas as capturas devem ter a mesma resolução.")
        tamanho = img.shape[1::-1]
        achou, pts = cantos(img)
        if achou:
            objetos.append(pontos_tabuleiro(args.quadrado))
            observados.append(pts)
            nomes.append(path)
    if len(nomes) < 15:
        raise ValueError(f"São necessárias 15 vistas válidas; encontradas {len(nomes)}.")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objetos, observados, tamanho, None, None)
    # K: fx/fy = distâncias focais em pixels; cx/cy = ponto principal em pixels.
    # dist: k1,k2,k3 representam distorção radial; p1,p2 distorção tangencial.
    # Ordem do OpenCV: k1,k2,p1,p2,k3. tvec tem a unidade do quadrado (metros).
    # RMS < 1 px é uma referência prática inicial, não garantia de segurança.
    # Robótica de precisão pode exigir < 0,5 px e validação independente de pose.
    linhas = []
    for path, obj, pts, r, t in zip(nomes, objetos, observados, rvecs, tvecs):
        reproj, _ = cv2.projectPoints(obj, r, t, K, dist)
        erro = float(np.sqrt(np.mean(np.sum((pts-reproj)**2, axis=2))))
        linhas.append({"imagem": path.name, "rms_px": erro})
    destino = RESULTADOS / "ex1"
    destino.mkdir(exist_ok=True)
    np.savez(destino / "calibracao.npz", K=K, dist=dist, tamanho=tamanho, quadrado=args.quadrado, padrao=PADRAO, unidade=args.unidade)
    csv_salvar(destino / "reprojecao.csv", linhas)
    media = float(np.mean([r["rms_px"] for r in linhas]))
    resumo = {"vistas": len(nomes), "K": K.tolist(), "dist": dist.ravel().tolist(),
              "rms_global_px": rms, "media_rms_por_imagem_px": media,
              "tempo_total_ms": (perf_counter()-inicio)*1000,
              "origem": str(args.pasta), "aresta_quadrado": args.quadrado, "unidade": args.unidade}
    json_salvar(destino / "metricas.json", resumo)
    print("K:\n", K, "\nk1,k2,p1,p2,k3:", dist.ravel(), f"\nErro médio: {media:.4f} px")
    img = ler(nomes[0])
    corrigida = cv2.undistort(img, K, dist)
    painel = np.hstack([img, corrigida])
    salvar(destino / "original_corrigida.jpg", painel)
    if not args.sem_janelas:
        cv2.imshow("Original | Corrigida", painel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def realidade_aumentada(args):
    global PADRAO
    dados = np.load(RESULTADOS / "ex1" / "calibracao.npz")
    PADRAO = tuple(dados["padrao"])
    K, dist = dados["K"], dados["dist"]
    s = float(dados["quadrado"])
    obj = pontos_tabuleiro(s)
    cubo = np.float32([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                       [0,0,-1],[1,0,-1],[1,1,-1],[0,1,-1]]) * s
    cap = abrir_video(args.fonte)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.inicio)
    writer = None
    linhas = []
    salvou_cubo = False
    try:
        while len(linhas) < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            if tuple(dados["tamanho"]) != frame.shape[1::-1]:
                raise ValueError("Use vídeo da mesma câmera e resolução da calibração.")
            if writer is None:
                writer = escritor(RESULTADOS / "ex1" / "cubo.avi", cap.get(cv2.CAP_PROP_FPS) or 30, frame.shape[1::-1])
            inicio = perf_counter()
            pose = {"rx":None,"ry":None,"rz":None,"tx":None,"ty":None,"tz":None}
            achou, pts = cantos(frame)
            if achou:
                sucesso, rvec, tvec = cv2.solvePnP(obj, pts, K, dist)
                if sucesso:
                    projetados, _ = cv2.projectPoints(cubo, rvec, tvec, K, dist)
                    p = np.int32(np.rint(projetados.reshape(-1, 2)))
                    cv2.polylines(frame, [p[:4]], True, (0,255,0), 2)
                    cv2.polylines(frame, [p[4:]], True, (0,0,255), 2)
                    for i in range(4):
                        cv2.line(frame, tuple(p[i]), tuple(p[i+4]), (255,0,0), 2)
                    print(f"Frame {len(linhas)+args.inicio}: rvec={rvec.ravel()} tvec({dados['unidade']})={tvec.ravel()}")
                    pose = dict(zip(pose, [float(v) for v in np.r_[rvec.ravel(),tvec.ravel()]]))
                    if not salvou_cubo:
                        salvar(RESULTADOS / "ex1" / "cubo.jpg", frame)
                        salvou_cubo = True
            else:
                print(f"Frame {len(linhas)+args.inicio}: tabuleiro não encontrado; pose indisponível.")
            linhas.append({"frame": len(linhas)+args.inicio, "tabuleiro": bool(achou), "ms": (perf_counter()-inicio)*1000, **pose})
            writer.write(frame)
            if not args.sem_janelas:
                cv2.imshow("Realidade aumentada", frame)
                if cv2.waitKey(1) & 255 == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    csv_salvar(RESULTADOS / "ex1" / "pose_tempos.csv", linhas)
    poses = [r for r in linhas if r["tabuleiro"]]
    tempos = [r["ms"] for r in linhas]
    json_salvar(RESULTADOS / "ex1" / "pose_resumo.json", {
        "frames": len(linhas), "frames_com_pose": len(poses),
        "taxa_deteccao": len(poses)/len(linhas) if linhas else 0,
        "tempo_medio_ms": float(np.mean(tempos)) if tempos else None,
        "tempo_mediano_ms": float(np.median(tempos)) if tempos else None,
        "unidade_translacao": str(dados["unidade"]),
        "origem": str(args.fonte), "inicio_frame": args.inicio})

if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("modo", choices=["capturar", "calibrar", "ar"])
    parser.add_argument("--fonte", default="0")
    parser.add_argument("--pasta", type=Path, default=DADOS / "tabuleiro")
    parser.add_argument("--quadrado", type=float, default=0.025, help="Aresta física medida em metros")
    parser.add_argument("--unidade", default="m", help="Use quadrados se o tamanho físico do dataset for desconhecido")
    parser.add_argument("--padrao", type=int, nargs=2, default=[7,6])
    parser.add_argument("--inicio", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--sem-janelas", action="store_true")
    args = parser.parse_args()
    PADRAO = tuple(args.padrao)
    {"capturar": capturar, "calibrar": calibrar, "ar": realidade_aumentada}[args.modo](args)
