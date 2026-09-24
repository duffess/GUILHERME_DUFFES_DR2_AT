"""Extrai cinco imagens urbanas espaçadas do vídeo público MOT15 TUD-Stadtmitte."""
from pathlib import Path
import shutil
import cv2
from comum import DADOS, MODELOS, abrir_video, salvar, json_salvar, baixar

def main():
    fonte = DADOS / "mot_pedestres.avi"
    destino = DADOS / "cenas_externas"
    destino.mkdir(parents=True, exist_ok=True)
    for anterior in destino.glob("mot15_tud_*.jpg"):
        anterior.unlink()
    # Três cenas KITTI de vias urbanas e dois quadros espaçados do MOT15
    # (caminho/calçada urbana). As imagens KITTI são amostras didáticas públicas.
    base = "https://csundergrad.science.uoit.ca/courses/csci3240u/latest/labs/data/kitti-samples/image_2"
    cenas = ["um_000032.png", "umm_000005.png", "uu_000010.png"]
    manifest = []
    for nome in cenas:
        src = baixar(f"{base}/{nome}", MODELOS / "cenas_ex4" / nome)
        dst = destino / nome
        shutil.copyfile(src, dst)
        manifest.append({"arquivo":nome,"dataset":"KITTI Road/Lane benchmark sample","origem":f"{base}/{nome}","categoria":"cena urbana externa"})

    cap = abrir_video(fonte)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [44, 134]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Falha lendo frame {idx} de {fonte}")
        path = destino / f"mot15_tud_{idx:04d}.jpg"
        salvar(path, frame)
        manifest.append({"arquivo":path.name,"frame":idx,"video":"dados/mot_pedestres.avi","dataset":"MOT15 TUD-Stadtmitte","categoria":"cena externa de pedestres"})
    cap.release()
    cv2.destroyAllWindows()
    json_salvar(destino / "ORIGEM.json", manifest)
    print(f"{len(manifest)} imagens extraídas para {destino}")

if __name__ == "__main__":
    main()
