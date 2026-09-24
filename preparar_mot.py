"""Baixa só TUD-Stadtmitte do ZIP público MOT15 usando HTTP Range (~poucos MB)."""
import io
import json
import struct
import subprocess
from zipfile import ZipFile
import cv2
from comum import DADOS, ler, escritor, json_salvar

URL = 'https://motchallenge.net/data/MOT15.zip'

class PartesZIP(io.RawIOBase):
    def __init__(self,total,partes):
        self.total,self.partes,self.pos = total,partes,0
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,n,whence=0):
        self.pos = n if whence==0 else self.pos+n if whence==1 else self.total+n
        return self.pos
    def read(self,n=-1):
        if n<0: n=self.total-self.pos
        for inicio,dados in self.partes:
            if inicio<=self.pos and self.pos+n<=inicio+len(dados):
                deslocamento=self.pos-inicio; self.pos+=n
                return dados[deslocamento:deslocamento+n]
        raise ValueError(f'Intervalo ZIP não disponível: {self.pos}, {n}')

def obter(intervalo,path):
    if not path.exists():
        subprocess.run(['curl.exe','-fL','--retry','2','-r',intervalo,URL,'-o',str(path)],check=True)
    return path.read_bytes()

def main():
    pasta=DADOS/'mot'; pasta.mkdir(exist_ok=True)
    # EOCD contém o offset do diretório central. Não baixa o ZIP de 1,22 GB inteiro.
    cauda=obter('-4194304',pasta/'central.bin')
    pos=cauda.rfind(b'PK\x05\x06')
    if pos<0: raise ValueError('Diretório central ZIP não encontrado')
    eocd=struct.unpack_from('<4s4H2LH',cauda,pos)
    total=eocd[6]+eocd[5]+22+eocd[7]
    inicio_cauda=total-len(cauda)
    partes=[(inicio_cauda,cauda)]
    arquivo=PartesZIP(total,partes)
    with ZipFile(arquivo) as z:
        membros=[m for m in z.infolist() if '/train/TUD-Stadtmitte/' in '/'+m.filename and not m.is_dir() and ('/img1/' in m.filename or '/gt/' in m.filename or m.filename.endswith('seqinfo.ini'))]
        if not membros: raise ValueError('Sequência ausente no ZIP')
        inicio=min(m.header_offset for m in membros)
        fim=max(m.header_offset+30+len(m.filename.encode())+len(m.extra)+m.compress_size+1024 for m in membros)
        dados=obter(f'{inicio}-{fim}',pasta/'sequencia.bin')
        partes.append((inicio,dados))
        for m in membros:
            relativo=m.filename.split('TUD-Stadtmitte/',1)[1]
            destino=pasta/relativo; destino.parent.mkdir(parents=True,exist_ok=True)
            destino.write_bytes(z.read(m))
    imagens=sorted((pasta/'img1').glob('*.jpg'))
    primeiro=ler(imagens[0])
    fps=25  # seqinfo.ini de TUD-Stadtmitte; validado abaixo.
    import configparser
    config=configparser.ConfigParser(); config.read(pasta/'seqinfo.ini')
    fps=float(config['Sequence']['frameRate'])
    writer=escritor(DADOS/'mot_pedestres.avi',fps,primeiro.shape[1::-1])
    try:
        for p in imagens: writer.write(ler(p))
    finally: writer.release()
    gt={str(n):[] for n in range(len(imagens))}
    for linha in (pasta/'gt'/'gt.txt').read_text().splitlines():
        v=[float(x) for x in linha.split(',')]
        if v[6] != 0:
            gt[str(int(v[0])-1)].append({'id':int(v[1]),'bbox':[v[2]-1,v[3]-1,v[4],v[5]]})
    json_salvar(DADOS/'mot_gt.json',gt)
    json_salvar(pasta/'ORIGEM.json',{'url':URL,'sequencia':'TUD-Stadtmitte','frames':len(imagens),'fps':fps,'gt':'Anotação humana oficial MOT15, split train usado apenas para avaliação; detectores não treinados aqui'})
    print(f'{len(imagens)} frames e GT preparados; fps={fps}.')

if __name__=='__main__': main()
