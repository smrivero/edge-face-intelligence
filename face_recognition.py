import cv2, time, json, threading
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data/face_template.npy'
MODEL = str(BASE / 'models/arcface.onnx')
W,H,PORT = 640,480,8765
SAMPLES,THRESHOLD = 8,0.45  # provisional: calibrar con impostores

face_detector = cv2.FaceDetectorYN_create(str(BASE/'models/yunet.onnx'), '', (W,H), 0.65,0.3,5000)
net = cv2.dnn.readNetFromONNX(MODEL)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# InsightFace ArcFace 112x112 reference landmarks
TARGET = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]],dtype=np.float32)
lock=threading.Lock()
condition=threading.Condition()
state={'jpeg':None,'seq':0,'enrolling':False,'vectors':[],'last_sample':0,'message':'Ready'}
template=np.load(DATA).astype(np.float32) if DATA.exists() else None
if template is not None: template /= np.linalg.norm(template)+1e-12

def embedding(img,face):
    # YuNet: eye1, eye2, nose, mouth1, mouth2
    p = np.array(face[4:14],dtype=np.float32).reshape(5,2)
    eyes = sorted(p[:2],key=lambda x:x[0])
    mouth = sorted(p[3:],key=lambda x:x[0])
    landmarks=np.array([eyes[0],eyes[1],p[2],mouth[0],mouth[1]],dtype=np.float32)
    matrix,_=cv2.estimateAffinePartial2D(landmarks,TARGET,method=cv2.LMEDS)
    if matrix is None: return None
    aligned=cv2.warpAffine(img,matrix,(112,112),borderValue=0)
    # blobFromImage swaps BGR -> RGB with swapRB=True
    blob=cv2.dnn.blobFromImage(aligned,1/127.5,(112,112),(127.5,127.5,127.5),swapRB=True,crop=False)
    net.setInput(blob)
    v=net.forward().reshape(-1).astype(np.float32)
    v/=np.linalg.norm(v)+1e-12
    return v

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global template
        if self.path=='/enroll':
            with lock:
                state['vectors']=[]
                state['enrolling']=True
                state['last_sample']=0
                state['message']='Enrollment started'
            return self.reply(200,{'status':'capturing','samples_required':SAMPLES})
        if self.path=='/delete':
            with lock:
                template=None
                state['enrolling']=False
                state['vectors']=[]
                DATA.unlink(missing_ok=True)
                state['message']='Enrollment deleted'
            return self.reply(200,{'status':'deleted'})
        self.reply(404,{'error':'not found'})
    def reply(self,status,obj):
        b=json.dumps(obj).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path=='/status':
            with lock:
                result={'enrolled':template is not None,'enrolling':state['enrolling'],'samples':len(state['vectors']),'message':state['message']}
            return self.reply(200,result)
        if self.path=='/':
            page=b'<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n<title>Edge Face Intelligence | Face Recognition</title>\n<style>\nbody{font:16px Arial,sans-serif;background:#151515;color:#eee;text-align:center;margin:20px}\nbutton{padding:12px 20px;margin:6px;border:0;border-radius:7px;cursor:pointer;font-size:15px}\nbutton:disabled{opacity:.45;cursor:default}\n#register{background:#76b900;color:#111}\n#delete{background:#444;color:white}\n#status{margin:18px auto;padding:15px;max-width:650px;background:#242424;border-radius:8px}\n#progress{width:100%;max-width:600px;height:18px;accent-color:#76b900}\n#video{width:100%;max-width:1280px}\n.muted{color:#aaa;font-size:14px}\n</style>\n</head>\n<body>\n<h2>Jetson | Face Recognition + Depth</h2>\n<p class="muted">YuNet + ArcFace | RealSense D435</p>\n<div id="status">\n<h3 id="title">Consultando estado...</h3>\n<p id="message"></p>\n<progress id="progress" value="0" max="8"></progress>\n<p id="counter"></p>\n</div>\n<button id="register" onclick="enroll()">Registrar mi cara</button>\n<button id="delete" onclick="erase()">Eliminar registro</button>\n<p class="muted">Las 8 muestras se capturan autom\xc3\xa1ticamente.</p>\n<img id="video" src="/stream.mjpg">\n<script>\nlet pending=false;\nasync function request(path){\n  const r=await fetch(path,{method:\'POST\'});\n  if(!r.ok)throw Error(\'HTTP \'+r.status);\n  return r.json();\n}\nasync function enroll(){\n  if(!confirm(\'\xc2\xbfRegistrar nuevamente tu cara?\'))return;\n  try{\n    await request(\'/enroll\');\n    pending=true;\n    await update();\n  }catch(e){alert(\'Error: \'+e.message)}\n}\nasync function erase(){\n  if(!confirm(\'\xc2\xbfEliminar el registro biom\xc3\xa9trico?\'))return;\n  try{\n    await request(\'/delete\');\n    pending=false;\n    await update();\n  }catch(e){alert(\'Error: \'+e.message)}\n}\nasync function update(){\n  try{\n    const r=await fetch(\'/status\',{cache:\'no-store\'});\n    const s=await r.json();\n    const n=s.samples||0;\n    document.getElementById(\'progress\').value=n;\n    document.getElementById(\'counter\').textContent=n+\' / 8 muestras\';\n    document.getElementById(\'message\').textContent=s.message||\'\';\n    let title;\n    if(s.enrolling){\n      title=\'Registrando cara...\';\n    }else if(s.enrolled){\n      title=\'Registro completado\';\n      pending=false;\n    }else{\n      title=\'No hay identidad registrada\';\n      pending=false;\n    }\n    document.getElementById(\'title\').textContent=title;\n    document.getElementById(\'register\').disabled=s.enrolling;\n    document.getElementById(\'delete\').disabled=s.enrolling;\n    document.getElementById(\'progress\').style.display=\n      (s.enrolling||s.enrolled)?\'inline-block\':\'none\';\n    document.getElementById(\'counter\').style.display=\n      (s.enrolling||s.enrolled)?\'block\':\'none\';\n  }catch(e){\n    document.getElementById(\'title\').textContent=\'Sin conexi\xc3\xb3n\';\n  }\n}\nupdate();\nsetInterval(update,500);\n</script>\n</body>\n</html>'
            self.send_response(200)
            self.send_header('Content-Type','text/html')
            self.send_header('Content-Length',str(len(page)))
            self.end_headers()
            return self.wfile.write(page)
        if self.path!='/stream.mjpg': return self.reply(404,{'error':'not found'})
        self.send_response(200)
        self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        seq=-1
        try:
            while True:
                with condition:
                    condition.wait_for(lambda: state['seq']!=seq,timeout=5)
                    jpg=state['jpeg'];seq=state['seq']
                if jpg:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpg)).encode()+b'\r\n\r\n'+jpg+b'\r\n')
        except (BrokenPipeError,ConnectionResetError,OSError): pass
    def log_message(self,*args): pass

camera=rs.pipeline();config=rs.config()
config.enable_stream(rs.stream.color,W,H,rs.format.bgr8,30)
config.enable_stream(rs.stream.depth,W,H,rs.format.z16,30)
align=rs.align(rs.stream.color)
camera.start(config)
server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
print(f'Listening at http://127.0.0.1:{PORT}; template={DATA}',flush=True)
last=time.perf_counter();fps=0
try:
    while True:
        frames=align.process(camera.wait_for_frames(5000))
        cf,df=frames.get_color_frame(),frames.get_depth_frame()
        if not cf or not df: continue
        img=np.asarray(cf.get_data()).copy()
        raw=np.asarray(df.get_data())
        depth_mm=raw.astype(np.float32)*df.get_units()*1000
        _,faces=face_detector.detect(img)
        count=0 if faces is None else len(faces)
        elapsed_ms=0
        with lock:
            enrolling=state['enrolling']
        if count==1:
            face=faces[0]
            x,y,w,h=face[:4].astype(int)
            start=time.perf_counter()
            vec=embedding(img,face)
            elapsed_ms=(time.perf_counter()-start)*1000
            label='UNKNOWN'
            if vec is not None:
                with lock:
                    if enrolling and time.monotonic()-state['last_sample']>.3:
                        state['vectors'].append(vec)
                        state['last_sample']=time.monotonic()
                        state['message']=f"Capturing {len(state['vectors'])}/{SAMPLES}"
                        if len(state['vectors'])>=SAMPLES:
                            avg=np.mean(state['vectors'],axis=0)
                            avg/=np.linalg.norm(avg)+1e-12
                            DATA.parent.mkdir(parents=True,exist_ok=True)
                            np.save(DATA,avg)
                            DATA.chmod(0o600)
                            template=avg
                            state['enrolling']=False
                            state['message']='Enrollment complete'
                    enrolled=template.copy() if template is not None else None
                    ongoing=state['enrolling']
                    samples=len(state['vectors'])
                if ongoing:
                    label=f'ENROLLING {samples}/{SAMPLES}'
                elif enrolled is not None:
                    similarity=float(np.dot(vec,enrolled))
                    label=f"{'MATCH' if similarity>=THRESHOLD else 'UNKNOWN'} sim={similarity:.3f}"
                else: label='NOT ENROLLED'
            cx=int(np.clip(x+w//2,0,W-1));cy=int(np.clip(y+h//2,0,H-1))
            roi=depth_mm[max(0,cy-12):min(H,cy+12),max(0,cx-12):min(W,cx+12)]
            valid=roi[(roi>250)&(roi<3000)]
            distance=f'{np.median(valid):.0f}mm' if len(valid) else 'N/A'
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,f'{label} {distance}',(max(0,x),max(25,y-10)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
        elif enrolling:
            with lock: state['message']='Enrollment requires exactly one face'
        now=time.perf_counter();fps=.9*fps+.1/(max(now-last,.001));last=now
        cv2.putText(img,f'Faces {count} | FPS {fps:.1f} | ArcFace {elapsed_ms:.1f}ms',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.53,(0,255,0),2)
        depth_vis=cv2.applyColorMap(cv2.convertScaleAbs(depth_mm,alpha=255/2500),cv2.COLORMAP_JET)
        depth_vis[raw==0]=0
        ok,buffer=cv2.imencode('.jpg',np.hstack((img,depth_vis)),[cv2.IMWRITE_JPEG_QUALITY,80])
        if ok:
            with condition:
                state['jpeg']=buffer.tobytes();state['seq']+=1;condition.notify_all()
except KeyboardInterrupt: print('Stopping')
finally:
    server.shutdown();server.server_close();camera.stop()
