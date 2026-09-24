"""Rubricas CNN: treino do zero, regularização, augmentation, PCA e ORB.

Dados sintéticos de setas esquerda/direita: flip troca o rótulo corretamente.
Experimento didático, NÃO demonstra acurácia em trânsito real.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from time import perf_counter
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from comum import RESULTADOS, json_salvar, salvar

def amostras(n, seed, amplo):
    rng=np.random.default_rng(seed)
    X=[]; y=[]
    for i in range(n):
        label=i%2
        img=np.full((64,64,3),rng.integers(0,60),np.uint8)
        pontos=np.int32([[12,25],[35,25],[35,15],[53,32],[35,49],[35,39],[12,39]])
        if label==0: pontos[:,0]=63-pontos[:,0]
        cv2.fillPoly(img,[pontos],tuple(int(x) for x in rng.integers(140,255,3)))
        ang=rng.uniform(-25,25) if amplo else rng.uniform(-5,5)
        escala=rng.uniform(.7,1.15) if amplo else rng.uniform(.95,1.05)
        m=cv2.getRotationMatrix2D((32,32),ang,escala)
        img=cv2.warpAffine(img,m,(64,64),borderMode=cv2.BORDER_REFLECT)
        X.append(img); y.append(label)
    return np.array(X,dtype=np.float32)/255,np.array(y)

def aumentar(X,y):
    rng=np.random.default_rng(91)
    novas=[]; rotulos=[]
    for img,label in zip(X,y):
        m=cv2.getRotationMatrix2D((32,32),rng.uniform(-25,25),rng.uniform(.7,1.15))
        novo=cv2.warpAffine(img,m,(64,64),borderMode=cv2.BORDER_REFLECT)
        if rng.random()<.5:
            novo=cv2.flip(novo,1)
            label=1-label  # Esquerda <-> direita, não manter rótulo incorreto.
        novas.append(novo); rotulos.append(label)
    return np.concatenate([X,novas]),np.concatenate([y,rotulos])

def main():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.keras.utils.set_random_seed(42)
    out=RESULTADOS/'cnn'; out.mkdir(exist_ok=True)
    X,y=amostras(400,1,False)
    V,v=amostras(160,2,True)
    T,t=amostras(160,3,True)
    # Seeds distintos: teste nunca participa de seleção de época/PCA/scaler.
    salvar(out/'amostras.jpg',np.hstack([(a*255).astype(np.uint8) for a in T[:8]]))
    res=[]; historias={}
    extrator=None
    for nome,regularizar,aug in [('baseline',False,False),('regularizada',True,False),('augmentation',True,True)]:
        tf.keras.utils.set_random_seed(42)
        camadas=[tf.keras.layers.Input((64,64,3))]
        for filtros in (16,32):
            camadas += [tf.keras.layers.Conv2D(filtros,3,activation='relu'),tf.keras.layers.MaxPooling2D()]
            if regularizar: camadas += [tf.keras.layers.BatchNormalization()]
        camadas += [tf.keras.layers.Flatten(),tf.keras.layers.Dense(32,activation='relu',name='features')]
        if regularizar: camadas += [tf.keras.layers.Dropout(.4)]
        camadas += [tf.keras.layers.Dense(2,activation='softmax')]
        modelo=tf.keras.Sequential(camadas)
        modelo.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        tx,ty=aumentar(X,y) if aug else (X,y)
        inicio=perf_counter()
        h=modelo.fit(tx,ty,validation_data=(V,v),epochs=8,batch_size=32,verbose=0)
        treino_ms=(perf_counter()-inicio)*1000
        scores=modelo.predict(T,verbose=0)
        acuracia=float(np.mean(scores.argmax(1)==t))
        inicio=perf_counter(); modelo(T,training=False).numpy(); ms=(perf_counter()-inicio)*1000/len(T)
        res.append(dict(modelo=nome,acuracia_teste=acuracia,ms_por_imagem_em_batch=ms,treino_ms=treino_ms,
                        gap_treino_validacao=h.history['accuracy'][-1]-h.history['val_accuracy'][-1]))
        historias[nome]=h.history
        # Extratora do modelo com augmentation, escolha fixada antes do teste.
        if aug: extrator=tf.keras.Model(modelo.input,modelo.get_layer('features').output)
        print(res[-1])
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for nome,h in historias.items():
        ax[0].plot(h['accuracy'],label=nome+' treino'); ax[0].plot(h['val_accuracy'],'--',label=nome+' val')
        ax[1].plot(h['loss'],label=nome+' treino'); ax[1].plot(h['val_loss'],'--',label=nome+' val')
    for a in ax: a.legend(fontsize=6); a.set_xlabel('época'); a.grid()
    ax[0].set_ylabel('acurácia'); ax[1].set_ylabel('loss'); fig.tight_layout(); fig.savefig(out/'curvas.png',dpi=130); plt.close(fig)
    orb=cv2.ORB_create(nfeatures=100,edgeThreshold=5,patchSize=15)
    def orb_features(images):
        saida=[]
        for img in images:
            _,desc=orb.detectAndCompute(cv2.cvtColor((img*255).astype(np.uint8),cv2.COLOR_RGB2GRAY),None)
            # Agregação de descritores binários, fixa em 256 dimensões por imagem.
            saida.append(np.unpackbits(desc,axis=1).mean(0) if desc is not None else np.zeros(256))
        return np.array(saida)
    separacao=[]
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for a,nome,treino,teste in [(ax[0],'CNN',extrator.predict(X,verbose=0),extrator.predict(T,verbose=0)),
                               (ax[1],'ORB',orb_features(X),orb_features(T))]:
        scaler=StandardScaler().fit(treino)
        treino=scaler.transform(treino); teste=scaler.transform(teste)
        pca=PCA(n_components=2,random_state=42).fit(treino)
        proj=pca.transform(teste)
        clf=LogisticRegression(max_iter=1000).fit(treino,y)
        acc=float(clf.score(teste,t))
        sil=float(silhouette_score(proj,t))
        separacao.append(dict(features=nome,acuracia_linear_teste=acc,silhouette_pca=sil))
        a.scatter(proj[:,0],proj[:,1],c=t,cmap='coolwarm',s=12); a.set_title(f'{nome}: silhouette={sil:.3f}'); a.set_xlabel('PC1'); a.set_ylabel('PC2')
    fig.tight_layout(); fig.savefig(out/'pca_cnn_orb.png',dpi=140); plt.close(fig)
    json_salvar(out/'metricas.json',{'dados':'Setas sintéticas: 400 treino, 160 validação, 160 teste; seeds 1/2/3', 'experimentos':res,'separabilidade':separacao,
                'delta_augmentation_pp':100*(res[2]['acuracia_teste']-res[1]['acuracia_teste'])})
    json_salvar(out/'historicos.json',historias)
    # Overfitting: perda de treino cai enquanto validação estagna/piora.
    # Gap de acurácia isolado não prova overfitting; inspecionar ambas as curvas.
    # Melhoria é o delta medido, pode ser negativo. O teste não seleciona modelo.

if __name__=='__main__': main()
