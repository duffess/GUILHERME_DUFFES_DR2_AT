"""Utilidades de arquivos, evidências e download. Fontes em FONTES.md."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import subprocess
import sys
from time import perf_counter
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DADOS = ROOT / "dados"
MODELOS = ROOT / "modelos"
RESULTADOS = ROOT / "resultados"
for pasta in (DADOS, MODELOS, RESULTADOS):
    pasta.mkdir(exist_ok=True)

def ler(path, modo=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(path, np.uint8), modo)
    if img is None:
        raise ValueError(f"Imagem inválida: {path}")
    return img

def salvar(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"Falha ao salvar {path}")
    buffer.tofile(path)

def json_salvar(path, dados):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dados, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def csv_salvar(path, linhas):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if linhas:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(linhas)

def baixar(url, destino):
    """curl usa o repositório de certificados do Windows; TLS permanece verificado."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists():
        temporario = destino.with_suffix(destino.suffix + ".part")
        print(f"Baixando {destino.name}...", flush=True)
        subprocess.run(["curl.exe" if sys.platform == "win32" else "curl", "-fL", "--retry", "2",
                        "--connect-timeout", "30", url, "-o", str(temporario)], check=True)
        temporario.replace(destino)
    return destino

def fonte_video(texto):
    return int(texto) if str(texto).isdigit() else str(texto)

def abrir_video(fonte):
    cap = cv2.VideoCapture(fonte_video(fonte))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Não foi possível abrir a câmera/vídeo: {fonte}")
    return cap

def escritor(path, fps, tamanho):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, tamanho)
    if not writer.isOpened():
        raise RuntimeError(f"Falha ao criar vídeo {path}")
    return writer

def ambiente():
    dados = {"python": sys.version, "sistema": platform.platform(), "cpu": platform.processor(),
             "opencv": cv2.__version__, "numpy": np.__version__, "threads_opencv": cv2.getNumThreads()}
    json_salvar(RESULTADOS / "ambiente.json", dados)
    return dados

def medir(funcao, repeticoes=10):
    funcao()  # warm-up excluído
    tempos = []
    for _ in range(repeticoes):
        inicio = perf_counter()
        resultado = funcao()
        tempos.append((perf_counter()-inicio)*1000)
    return resultado, float(np.mean(tempos))
