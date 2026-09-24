"""DeepLabV3 e HSV nas mesmas cinco cenas externas."""
import argparse
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from comum import DADOS, MODELOS, RESULTADOS, baixar, ler, salvar, csv_salvar, json_salvar

def main(args):
    import torch
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
    from torchvision.transforms.functional import to_pil_image
    torch.set_num_threads(4)
    pesos = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    path = baixar(pesos.url, MODELOS / "deeplabv3.pth")
    modelo = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=21, aux_loss=True)
    modelo.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    modelo.eval()
    cores = np.random.default_rng(42).integers(40,255,(21,3),dtype=np.uint8)
    cores[0] = 0
    imagens = sorted(p for p in args.pasta.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg"))[:5]
    if len(imagens)<5:
        raise ValueError("São necessárias pelo menos cinco imagens externas.")
    linhas, tempos = [], []
    for path in imagens:
        img = ler(path)
        rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        inicio = perf_counter()
        tensor = pesos.transforms()(to_pil_image(rgb)).unsqueeze(0)
        with torch.inference_mode():
            pred = modelo(tensor)['out'].argmax(1)[0].cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred,img.shape[1::-1],interpolation=cv2.INTER_NEAREST)
        ms = (perf_counter()-inicio)*1000
        mapa = cores[pred]
        sobreposta = cv2.addWeighted(img,.6,mapa,.4,0)
        inicio = perf_counter()
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv,(35,40,40),(85,255,255))
        hsv_ms = (perf_counter()-inicio)*1000
        verde = img.copy()
        verde[mascara>0] = (0,255,0)
        verde = cv2.addWeighted(img,.6,verde,.4,0)
        for classe in np.unique(pred):
            percentual = float(np.mean(pred==classe)*100)
            nome = pesos.meta['categories'][int(classe)]
            print(f"{path.name}: {nome}: {percentual:.2f}%")
            linhas.append({"imagem":path.name,"classe":nome,"area_percentual":percentual})
        tempos.append({"imagem":path.name,"deeplab_ms":ms,"hsv_ms":hsv_ms})
        painel = np.hstack([sobreposta,verde])
        cv2.putText(painel,"DeepLabV3 | HSV verde (cor, nao classe)",(10,25),0,.6,(255,255,255),2)
        salvar(RESULTADOS / "ex4" / f"{path.stem}_painel.jpg",painel)
        salvar(RESULTADOS / "ex4" / f"{path.stem}_classes.png",pred)
        salvar(RESULTADOS / "ex4" / f"{path.stem}_cores.png",mapa)
        if not args.sem_janelas:
            cv2.imshow("Semantica | HSV",painel)
            cv2.waitKey(0)
    cv2.destroyAllWindows()
    csv_salvar(RESULTADOS / "ex4" / "areas.csv",linhas)
    csv_salvar(RESULTADOS / "ex4" / "tempos.csv",tempos)
    json_salvar(RESULTADOS / "ex4" / "legenda.json", {n:cores[i].tolist() for i,n in enumerate(pesos.meta['categories'])})
    # HSV é barato, explicável e depende de luz/cor. Não reconhece categorias.
    # DeepLab usa contexto aprendido, mas custa mais e pode errar fora do domínio.
    # Estes pesos possuem 21 classes VOC: NÃO distinguem pista nem calçada.
    # Background não significa área trafegável. Para autonomia urbana, treinar
    # com classes apropriadas (ex.: Cityscapes) e validar mIoU por classe.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pasta",type=Path,default=DADOS / "cenas_externas")
    parser.add_argument("--sem-janelas",action="store_true")
    main(parser.parse_args())
