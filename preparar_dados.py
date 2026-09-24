"""Dados públicos para demonstração e tabuleiro SINTÉTICO para teste técnico."""
import argparse
import json
import shutil
import tarfile
from zipfile import ZipFile
import cv2
import numpy as np
from comum import DADOS, MODELOS, baixar, salvar, json_salvar, escritor

def publicos():
    arquivo = baixar("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", DADOS / "imagenette.tgz")
    base = DADOS / "imagenette2-160"
    if not base.exists():
        with tarfile.open(arquivo) as tar:
            # Somente arquivos regulares da validação, dentro do diretório esperado.
            for membro in tar.getmembers():
                if membro.isfile() and "/val/" in membro.name and ".." not in membro.name.split("/"):
                    destino = DADOS / membro.name
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(membro) as entrada, destino.open("wb") as saida:
                        shutil.copyfileobj(entrada, saida)
    rotulos = baixar("https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json", MODELOS / "classes.json")
    classes = json.loads(rotulos.read_text())
    inverso = {v[0]: int(k) for k, v in classes.items()}
    manifesto = [{"arquivo": str(sorted(p.glob("*.JPEG"))[0].relative_to(DADOS)),
                  "classe": inverso[p.name], "synset": p.name} for p in sorted((base / "val").iterdir()) if p.is_dir()]
    json_salvar(DADOS / "classificacao.json", manifesto)
    arquivo = baixar("https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip", DADOS / "PennFudanPed.zip")
    if not (DADOS / "PennFudanPed").exists():
        with ZipFile(arquivo) as z:
            z.extractall(DADOS)
    # Vídeo público curto do OpenCV, utilizado só como demonstração de detecção.
    baixar("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi", DADOS / "pedestres_publico.avi")
    print("Dados públicos preparados; não são capturas do ambiente do aluno.")

def modelos_deteccao():
    baixar("https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg", MODELOS / "yolov4-tiny.cfg")
    baixar("https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights", MODELOS / "yolov4-tiny.weights")
    baixar("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", MODELOS / "coco.names")
    arquivo = baixar("https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz", MODELOS / "ssd.tar.gz")
    destino = MODELOS / "ssd_mobilenet_v2.pb"
    if not destino.exists():
        with tarfile.open(arquivo) as tar:
            membro = next(m for m in tar.getmembers() if m.name.endswith("/frozen_inference_graph.pb"))
            with tar.extractfile(membro) as entrada, destino.open("wb") as saida:
                shutil.copyfileobj(entrada, saida)
    baixar("https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt", MODELOS / "ssd_mobilenet_v2.pbtxt")

def tabuleiro_demo():
    pasta = DADOS / "tabuleiro_sintetico"
    pasta.mkdir(exist_ok=True)
    K = np.float64([[720,0,320],[0,715,240],[0,0,1]])
    dist = np.float64([-0.18, 0.04, 0.001, -0.001, 0.0])
    s = 0.025
    yy, xx = np.indices((480,640), dtype=np.float32)
    grade = np.stack([xx, yy], axis=-1).reshape(-1,1,2)
    origem = cv2.undistortPoints(grade, K, dist, P=K).reshape(480,640,2)
    def renderizar(rvec, tvec):
        img = np.full((480,640,3), 180, np.uint8)
        for y in range(-1, 6):
            for x in range(-1, 7):
                obj = np.float32([[x,y,0],[x+1,y,0],[x+1,y+1,0],[x,y+1,0]])*s
                pts, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
                cor = 245 if (x+y)%2 else 15
                cv2.fillConvexPoly(img, np.int32(np.rint(pts)), (cor,cor,cor), cv2.LINE_AA)
        return cv2.remap(img, origem[:,:,0], origem[:,:,1], cv2.INTER_LINEAR, borderValue=(180,180,180))
    rng = np.random.default_rng(17)
    for i in range(20):
        r = np.float64([rng.uniform(-.4,.4), rng.uniform(-.4,.4), rng.uniform(-.15,.15)])
        t = np.float64([rng.uniform(-.085,-.045), rng.uniform(-.075,-.035), rng.uniform(.50,.70)])
        salvar(pasta / f"vista_{i:02}.png", renderizar(r,t))
    video = escritor(DADOS / "tabuleiro_sintetico.avi", 20, (640,480))
    try:
        for i in range(80):
            a = i/30
            r = np.float64([.2*np.sin(a), .2*np.cos(a), .06*np.sin(a)])
            t = np.float64([-.065, -.055, .6+.03*np.sin(a)])
            video.write(renderizar(r,t))
    finally:
        video.release()
    json_salvar(pasta / "ORIGEM.json", {"tipo": "SINTETICO - não calibra uma câmera física", "K_real": K.tolist(), "dist_real": dist.tolist(), "quadrado_m": s})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tipo", choices=["publicos", "detectores", "tabuleiro", "todos"])
    args = parser.parse_args()
    for nome, funcao in [("publicos", publicos), ("detectores", modelos_deteccao), ("tabuleiro", tabuleiro_demo)]:
        if args.tipo in (nome, "todos"):
            funcao()
