"""MobileNetV2 com os MESMOS pesos no Keras e OpenCV DNN (exportação ONNX)."""
import argparse
import json
import os
import subprocess
import sys
from time import perf_counter
import cv2
import numpy as np
import psutil
from comum import DADOS, MODELOS, RESULTADOS, ROOT, baixar, ler, salvar, json_salvar

def blob(imagem):
    # RGB [-1,1], exatamente o preprocess_input do MobileNetV2.
    return cv2.dnn.blobFromImage(imagem, 1/127.5, (224,224), (127.5,127.5,127.5), swapRB=True, crop=False)

def carregar_dnn():
    net = cv2.dnn.readNetFromONNX(np.fromfile(MODELOS / "mobilenetv2.onnx", np.uint8))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def classificar(net, imagem):
    net.setInput(blob(imagem))
    return net.forward().ravel()

def exportar():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    import tf2onnx
    tf.config.threading.set_intra_op_parallelism_threads(4)
    pesos = baixar("https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5", MODELOS / "mobilenetv2.h5")
    model = tf.keras.applications.MobileNetV2(weights=None)
    model.load_weights(pesos)
    model.save(MODELOS / "mobilenetv2.keras")
    assinatura = [tf.TensorSpec((1,224,224,3), tf.float32, name="entrada")]
    tf2onnx.convert.from_keras(model, input_signature=assinatura, opset=13,
                             inputs_as_nchw=["entrada:0"], output_path=str(MODELOS / "mobilenetv2.onnx"))
    json_salvar(MODELOS / "mobilenet_info.json", {"parametros": model.count_params(), "origem": "MobileNetV2 ImageNet, mesmos pesos exportados de Keras para ONNX"})

def avaliar(backend):
    # Cada backend executa em processo isolado. RSS absoluto inclui runtime e modelo.
    cv2.setNumThreads(4)
    processo = psutil.Process()
    manifesto = json.loads((DADOS / "classificacao.json").read_text())
    if len(manifesto) < 10 or len({x["classe"] for x in manifesto}) < 10:
        raise ValueError("Forneça ao menos 10 imagens de 10 categorias distintas.")
    if backend == "keras":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        modelo = tf.keras.models.load_model(MODELOS / "mobilenetv2.keras")
        @tf.function(input_signature=[tf.TensorSpec((1,224,224,3), tf.float32)])
        def inferencia(x):
            return modelo(x, training=False)
        def executar(x):
            return inferencia(x.transpose(0,2,3,1)).numpy().ravel()
    else:
        modelo = carregar_dnn()
        def executar(x):
            modelo.setInput(x)
            return modelo.forward().ravel()
    dummy = blob(ler(DADOS / manifesto[0]["arquivo"]))
    for _ in range(5):
        executar(dummy)
    tempos, previsoes, memoria, linhas = [], [], [], []
    classes = json.loads((MODELOS / "classes.json").read_text())
    for item in manifesto:
        imagem = ler(DADOS / item["arquivo"])
        entrada = blob(imagem)
        # Pré-processamento fora do cronômetro nos dois backends; 10 repetições.
        for _ in range(10):
            inicio = perf_counter()
            scores = executar(entrada)
            tempos.append((perf_counter()-inicio)*1000)
            memoria.append(processo.memory_info().rss / 1024**2)
        top = np.argsort(scores)[-3:][::-1]
        previsoes.append(scores.tolist())
        linhas.append({"arquivo": item["arquivo"], "verdade": item["classe"], "predito": int(top[0])})
        imagem = cv2.resize(imagem, (640,480))
        for i, classe in enumerate(top):
            texto = f"{classes[str(classe)][1]}: {scores[classe]:.1%}"
            cv2.putText(imagem, texto, (10,30+30*i), 0, .65, (0,255,0), 2)
        salvar(RESULTADOS / "ex2" / backend / (str(item["classe"])+".jpg"), imagem)
    resultado = {"backend": backend, "latencia_ms": float(np.mean(tempos)),
                 "rss_max_amostrado_mb": max(memoria), "acuracia_top1": float(np.mean([r["verdade"] == r["predito"] for r in linhas])),
                 "imagens": linhas, "scores": previsoes, "repeticoes": len(tempos),
                 "memoria_definicao": "Máximo RSS amostrado após inferências; inclui runtime, não é pico contínuo nem memória exclusiva do modelo"}
    json_salvar(RESULTADOS / "ex2" / f"{backend}.json", resultado)

def comparar():
    for backend in ("opencv", "keras"):
        subprocess.run([sys.executable, str(ROOT / "ex2_classificacao.py"), "avaliar", "--backend", backend], check=True)
    resultados = [json.loads((RESULTADOS / "ex2" / f"{b}.json").read_text(encoding="utf-8")) for b in ("opencv", "keras")]
    erro = float(np.max(np.abs(np.array(resultados[0]["scores"])-np.array(resultados[1]["scores"]))))
    print("Backend | Latencia ms | RSS MB | Acuracia top-1")
    for r in resultados:
        print(f"{r['backend']:7} | {r['latencia_ms']:11.3f} | {r['rss_max_amostrado_mb']:7.1f} | {r['acuracia_top1']:.1%}")
    print(f"Maior diferença entre probabilidades: {erro:.8f}")
    if erro > 0.001:
        raise RuntimeError("Backends não equivalentes: verifique exportação e normalização.")
    json_salvar(RESULTADOS / "ex2" / "equivalencia.json", {"max_erro_probabilidade": erro})
    # DNN é preferível quando suas operações são suportadas e as medições mostram
    # menor latência/memória. Keras favorece treinamento e flexibilidade. Não se
    # deve inferir consumo de 5 W a partir do RSS ou FPS medidos neste desktop.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modo", choices=["exportar", "avaliar", "comparar"])
    parser.add_argument("--backend", choices=["opencv", "keras"], default="opencv")
    args = parser.parse_args()
    if args.modo == "avaliar":
        avaliar(args.backend)
    elif args.modo == "exportar":
        exportar()
    else:
        comparar()
