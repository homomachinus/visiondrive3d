"""
Autonomous Driving – Stylized View v4  (GLB 3D Model Edition)
=============================================================
Fix utama dari v3:
  1. Homography-based projection  — bbox bottom-centre di original frame
     di-warp langsung ke canvas coordinate via perspective transform.
     Ini jauh lebih akurat karena memanfaatkan geometri perspektif nyata.
  2. Ukuran model 3D proporsional NYATA — pakai rasio bbox_height/frame_height
     sebagai proxy focal-length depth, bukan formula ad-hoc.
  3. Posisi model "duduk" di road — proj_y adalah titik bawah model,
     kalibrasi tepat ke road surface.
  4. Label di bawah model, tidak menutupi.
  5. Render GLB offscreen 512×512 → resize bicubic → alpha composite.

Dependensi:
    pip install ultralytics opencv-python torch pygltflib PyOpenGL PyOpenGL_accelerate
"""

import os, sys, math, struct, ctypes
import numpy as np
import cv2
import torch
from ultralytics import YOLO

try:
    from pygltflib import GLTF2
except ImportError:
    sys.exit("pip install pygltflib")

try:
    from OpenGL import GL
    from OpenGL.GL import shaders
except ImportError:
    sys.exit("pip install PyOpenGL PyOpenGL_accelerate")


# ══════════════════════════════════════════════════════════════════
# CONFIG — sesuaikan path
# ══════════════════════════════════════════════════════════════════
ROAD_YOLO_WEIGHTS      = r"D:\mobil\runs\segment\my-seg2\weights\best.pt"
YOLO11N_WEIGHTS        = "yolo11n.pt"
CAR_GLB_PATH           = "car.glb"
VIDEO_PATH             = "v3.mp4"
OUTPUT_PATH            = "autonomous_view_v4.mp4"

CONF_ROAD              = 0.30
CONF_VEHICLE           = 0.28
MASK_THRESH            = 0.40
TARGET_CLASSES         = {"car", "truck", "bus", "motorcycle"}

# Canvas colours
SKY_BGR                = (28, 28, 32)
ROAD_BGR               = (205, 205, 210)
LANE_BGR               = (0, 215, 255)
EGO_ACCENT             = (0, 220, 255)
HORIZON_FRAC           = 0.44      # fraction of frame height for vanishing point

# GLB render texture size (internal)
OGL_W, OGL_H           = 512, 512

# ── Homography source points (in original frame, normalised 0-1) ──
# These 4 points define the "road quad" in the camera image.
# Bottom-left, bottom-right, top-right (VP), top-left (VP)
# Tune these to match your dashcam perspective.
SRC_PTS = np.float32([
    [0.00, 1.00],   # bottom-left of road in frame
    [1.00, 1.00],   # bottom-right
    [0.62, 0.44],   # vanishing-point right
    [0.38, 0.44],   # vanishing-point left
])

# ── Homography destination points (in canvas, normalised 0-1) ──
# Maps the road quad to the full canvas (minus ego car area at bottom)
DST_PTS = np.float32([
    [0.00, 0.80],   # bottom-left of road on canvas
    [1.00, 0.80],   # bottom-right
    [0.60, 0.44],   # vanishing-point right on canvas
    [0.40, 0.44],   # vanishing-point left on canvas
])

# Homography matrix — computed once, reused every frame
_H_MAT = None  # initialised in main() after we know W, H


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ══════════════════════════════════════════════════════════════════
# OPENGL CONTEXT  (hidden GLUT window — works on Windows)
# ══════════════════════════════════════════════════════════════════
def init_gl_context(w, h):
    try:
        from OpenGL.GLUT import (glutInit, glutInitDisplayMode, glutInitWindowSize,
                                  glutCreateWindow, glutHideWindow,
                                  GLUT_RGBA, GLUT_DOUBLE, GLUT_DEPTH)
        glutInit([])
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(w, h)
        glutCreateWindow(b"offscreen")
        glutHideWindow()
        return True
    except Exception as e:
        print(f"[GLUT] {e}")
    try:
        from OpenGL.osmesa import OSMesaCreateContext, OSMesaMakeCurrent
        ctx = OSMesaCreateContext(GL.GL_RGBA, None)
        buf = (GL.GLubyte * (w * h * 4))()
        OSMesaMakeCurrent(ctx, buf, GL.GL_UNSIGNED_BYTE, w, h)
        return True
    except Exception as e:
        print(f"[OSMesa] {e}")
    return False


# ══════════════════════════════════════════════════════════════════
# GLB MESH LOADER
# ══════════════════════════════════════════════════════════════════
class GLBMesh:
    _COMP_FMT   = {5120:'b',5121:'B',5122:'h',5123:'H',5125:'I',5126:'f'}
    _COMP_COUNT = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}

    def __init__(self, path):
        self.vaos       = []
        self._gl_ready  = False
        gltf = GLTF2().load(path)
        # binary blob
        import base64
        buf = gltf.buffers[0]
        if buf.uri and buf.uri.startswith("data:"):
            self._bin = base64.b64decode(buf.uri.split(",",1)[1])
        elif buf.uri:
            self._bin = open(os.path.join(os.path.dirname(path), buf.uri),"rb").read()
        else:
            self._bin = bytes(gltf.binary_blob())

        self._prims = []
        for mesh in (gltf.meshes or []):
            for prim in mesh.primitives:
                pos_idx = getattr(prim.attributes,"POSITION",None)
                if pos_idx is None: continue
                verts   = self._acc(gltf, pos_idx)
                norm_idx= getattr(prim.attributes,"NORMAL",None)
                normals = self._acc(gltf,norm_idx) if norm_idx is not None \
                          else np.tile([0,1,0],(len(verts),1)).astype(np.float32)
                idx_arr = self._acc(gltf,prim.indices).astype(np.uint32).flatten() \
                          if prim.indices is not None \
                          else np.arange(len(verts),dtype=np.uint32)
                col = np.array([0.72,0.72,0.76,1.0],dtype=np.float32)
                if prim.material is not None and gltf.materials:
                    mat = gltf.materials[prim.material]
                    if mat.pbrMetallicRoughness and mat.pbrMetallicRoughness.baseColorFactor:
                        col = np.array(mat.pbrMetallicRoughness.baseColorFactor,dtype=np.float32)
                self._prims.append((verts,normals,idx_arr,col))

        # normalise to unit cube centred at origin
        if self._prims:
            all_v = np.concatenate([p[0] for p in self._prims])
            mn,mx = all_v.min(0),all_v.max(0)
            c  = (mn+mx)/2
            sc = 1.0/max((mx-mn).max(), 1e-6)
            self._prims = [((v-c)*sc,n,i,col) for v,n,i,col in self._prims]

    def _acc(self, gltf, idx):
        acc = gltf.accessors[idx]
        bv  = gltf.bufferViews[acc.bufferView]
        fmt = self._COMP_FMT[acc.componentType]
        n   = self._COMP_COUNT[acc.type]
        off = (bv.byteOffset or 0)+(acc.byteOffset or 0)
        raw = self._bin[off: off+acc.count*n*struct.calcsize(fmt)]
        arr = np.frombuffer(raw,dtype=np.dtype(fmt)).reshape(acc.count,n)
        return arr.astype(np.float32)

    def upload(self):
        if self._gl_ready: return
        for verts,normals,indices,col in self._prims:
            vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(vao)
            data = np.hstack([verts,normals]).astype(np.float32)
            vbo  = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER,vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER,data.nbytes,data,GL.GL_STATIC_DRAW)
            ebo  = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER,ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER,indices.nbytes,indices,GL.GL_STATIC_DRAW)
            s = 6*4
            GL.glVertexAttribPointer(0,3,GL.GL_FLOAT,False,s,ctypes.c_void_p(0))
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(1,3,GL.GL_FLOAT,False,s,ctypes.c_void_p(12))
            GL.glEnableVertexAttribArray(1)
            GL.glBindVertexArray(0)
            self.vaos.append((vao,len(indices),col.copy()))
        self._gl_ready = True

    def draw(self, prog, col_override=None):
        for vao,cnt,col in self.vaos:
            c = col_override if col_override is not None else col
            GL.glUniform4fv(GL.glGetUniformLocation(prog,"uBaseColor"),1,c)
            GL.glBindVertexArray(vao)
            GL.glDrawElements(GL.GL_TRIANGLES,cnt,GL.GL_UNSIGNED_INT,None)
        GL.glBindVertexArray(0)


# ══════════════════════════════════════════════════════════════════
# SHADER
# ══════════════════════════════════════════════════════════════════
_VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uModel;
out vec3 vNorm;
out vec3 vFrag;
void main(){
    vec4 wp = uModel*vec4(aPos,1.);
    vFrag = wp.xyz;
    vNorm = mat3(transpose(inverse(uModel)))*aNormal;
    gl_Position = uMVP*vec4(aPos,1.);
}
"""
_FRAG = """
#version 330 core
in vec3 vNorm; in vec3 vFrag;
uniform vec4 uBaseColor;
uniform vec3 uLightDir;
uniform vec3 uViewPos;
out vec4 FragColor;
void main(){
    vec3 N=normalize(vNorm);
    vec3 L=normalize(-uLightDir);
    vec3 H=normalize(L+normalize(uViewPos-vFrag));
    float a=0.38, d=max(dot(N,L),0.)*0.50, s=pow(max(dot(N,H),0.),48.)*0.22;
    FragColor=vec4(uBaseColor.rgb*(a+d)+s, uBaseColor.a);
}
"""

def build_shader():
    return shaders.compileProgram(
        shaders.compileShader(_VERT, GL.GL_VERTEX_SHADER),
        shaders.compileShader(_FRAG, GL.GL_FRAGMENT_SHADER))


# ══════════════════════════════════════════════════════════════════
# MATRIX HELPERS
# ══════════════════════════════════════════════════════════════════
def _persp(fovy, asp, n, f):
    t = math.tan(math.radians(fovy)/2)
    return np.array([
        [1/(asp*t),0,0,0],[0,1/t,0,0],
        [0,0,-(f+n)/(f-n),-2*f*n/(f-n)],[0,0,-1,0]
    ],dtype=np.float32).T

def _lookat(eye,at,up):
    f=(at-eye); f/=np.linalg.norm(f)
    r=np.cross(f,up); r/=np.linalg.norm(r)
    u=np.cross(r,f)
    M=np.eye(4,dtype=np.float32)
    M[0,:3]=r; M[1,:3]=u; M[2,:3]=-f
    T=np.eye(4,dtype=np.float32); T[:3,3]=-eye
    return M@T

def _scale(s):
    M=np.eye(4,dtype=np.float32); M[0,0]=M[1,1]=M[2,2]=s; return M

def _rot_y(deg):
    a=math.radians(deg); c,s=math.cos(a),math.sin(a)
    M=np.eye(4,dtype=np.float32)
    M[0,0]=c; M[0,2]=s; M[2,0]=-s; M[2,2]=c; return M

def _rot_x(deg):
    a=math.radians(deg); c,s=math.cos(a),math.sin(a)
    M=np.eye(4,dtype=np.float32)
    M[1,1]=c; M[1,2]=-s; M[2,1]=s; M[2,2]=c; return M


# ══════════════════════════════════════════════════════════════════
# CAR RENDERER  (offscreen FBO)
# ══════════════════════════════════════════════════════════════════
class CarRenderer:
    def __init__(self, glb_path, ogl_w=OGL_W, ogl_h=OGL_H):
        self.W, self.H = ogl_w, ogl_h
        self.mesh = GLBMesh(glb_path)
        self._ready = False

    def init_gl(self):
        self.mesh.upload()
        self.prog = build_shader()

        # FBO
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        self.tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D,0,GL.GL_RGBA,self.W,self.H,0,
                        GL.GL_RGBA,GL.GL_UNSIGNED_BYTE,None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D,self.tex,0)

        self.rbo = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,self.rbo)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER,GL.GL_DEPTH24_STENCIL8,self.W,self.H)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                     GL.GL_RENDERBUFFER,self.rbo)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA,GL.GL_ONE_MINUS_SRC_ALPHA)
        self._ready = True

    def render_rgba(self, depth_ratio, aspect_ratio=1.8) -> np.ndarray:
        """
        Render model 3D ke RGBA image (H×W×4), background transparan.
        depth_ratio: 0=dekat(merah), 1=jauh(hijau)
        aspect_ratio: lebar/tinggi model yang diinginkan
        """
        # Danger colour (merah=dekat, hijau=jauh)
        r = clamp((1.0-depth_ratio)*0.85, 0, 1)
        g = clamp(depth_ratio*0.70,       0, 1)
        col_override = np.array([r, g, 0.08, 1.0], dtype=np.float32)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0,0,self.W,self.H)
        GL.glClearColor(0,0,0,0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
        GL.glUseProgram(self.prog)

        # Kamera: sedikit dari atas-depan, sudut ~30°
        eye    = np.array([0.0, 0.55, 2.0], dtype=np.float32)
        center = np.array([0.0, 0.0,  0.0], dtype=np.float32)
        up     = np.array([0.0, 1.0,  0.0], dtype=np.float32)

        proj  = _persp(38.0, self.W/self.H, 0.01, 50.0)
        view  = _lookat(eye, center, up)
        # Rotasi Y 180° supaya tampak depan mobil menghadap kita
        model = _rot_y(180.0) @ _scale(0.88)

        mvp = proj @ view @ model
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.prog,"uMVP"),  1,False,mvp.flatten())
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.prog,"uModel"),1,False,model.flatten())

        ld = np.array([-0.3,-0.7,-0.5],dtype=np.float32)
        ld /= np.linalg.norm(ld)
        GL.glUniform3fv(GL.glGetUniformLocation(self.prog,"uLightDir"),1,ld)
        GL.glUniform3fv(GL.glGetUniformLocation(self.prog,"uViewPos"), 1,eye)

        self.mesh.draw(self.prog, col_override)

        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT,1)
        raw = GL.glReadPixels(0,0,self.W,self.H,GL.GL_RGBA,GL.GL_UNSIGNED_BYTE)
        img = np.frombuffer(raw,dtype=np.uint8).reshape(self.H,self.W,4)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
        return np.flipud(img)  # flip vertikal

    def composite_onto(self, canvas: np.ndarray,
                       proj_x: int, proj_y: int,
                       target_w: int, target_h: int,
                       depth_ratio: float):
        """
        Alpha-blend model 3D ke canvas OpenCV.
        proj_x, proj_y = titik TENGAH-BAWAH model di canvas.
        target_w, target_h = ukuran model di canvas (dari bbox calibration).
        """
        rgba = self.render_rgba(depth_ratio)

        # Crop area non-transparan supaya resize lebih efisien
        alpha = rgba[:,:,3]
        rows  = np.any(alpha>10, axis=1)
        cols  = np.any(alpha>10, axis=0)
        if not rows.any() or not cols.any():
            return
        r0,r1 = np.where(rows)[0][[0,-1]]
        c0,c1 = np.where(cols)[0][[0,-1]]
        cropped = rgba[r0:r1+1, c0:c1+1]

        # Jaga aspect ratio saat resize
        cr_h, cr_w = cropped.shape[:2]
        if cr_w == 0 or cr_h == 0:
            return
        scale_factor = min(target_w/cr_w, target_h/cr_h)
        new_w = max(6,  int(cr_w * scale_factor))
        new_h = max(10, int(cr_h * scale_factor))

        small = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Posisi di canvas (proj_y = bawah model)
        x0 = proj_x - new_w//2
        y0 = proj_y - new_h
        x1 = x0 + new_w
        y1 = proj_y

        # Clipping
        sx0 = max(0, -x0);       sy0 = max(0, -y0)
        cx0 = max(0, x0);        cy0 = max(0, y0)
        cx1 = min(canvas.shape[1], x1)
        cy1 = min(canvas.shape[0], y1)
        sx1 = sx0 + (cx1-cx0)
        sy1 = sy0 + (cy1-cy0)

        if cx0>=cx1 or cy0>=cy1 or sx0>=sx1 or sy0>=sy1:
            return

        sprite = small[sy0:sy1, sx0:sx1]
        a      = sprite[:,:,3:4].astype(np.float32)/255.0
        fg     = sprite[:,:,2::-1].astype(np.float32)   # RGBA→BGR
        roi    = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
        canvas[cy0:cy1, cx0:cx1] = (fg*a + roi*(1-a)).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
# HOMOGRAPHY-BASED PROJECTION
# ══════════════════════════════════════════════════════════════════
def build_homography(frame_w, frame_h):
    """
    Hitung matrix homography dari titik di original frame ke canvas.
    Kita warp bottom-centre bbox (kaki mobil di aspal) langsung ke canvas.
    """
    src = (SRC_PTS * np.array([frame_w, frame_h])).astype(np.float32)
    dst = (DST_PTS * np.array([frame_w, frame_h])).astype(np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def project_point(H, px, py):
    """Warp satu titik (px,py) dari frame space ke canvas space via homography H."""
    pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, H)
    return int(warped[0,0,0]), int(warped[0,0,1])


def compute_canvas_size(bbox_h_px, frame_h, base_size_frac=0.13):
    """
    Ukuran model di canvas langsung proporsional ke bbox height.
    bbox_h_px besar → mobil dekat → model besar.
    base_size_frac: ukuran di canvas saat bbox_h = frame_h (sangat dekat).
    """
    frac    = clamp(bbox_h_px / frame_h, 0.01, 0.70)
    car_h   = int(frame_h * base_size_frac * (frac / 0.30))  # normalised ke 30%
    car_h   = clamp(car_h, 12, int(frame_h * 0.35))
    return car_h


# ══════════════════════════════════════════════════════════════════
# ROAD HELPERS
# ══════════════════════════════════════════════════════════════════
def get_road_mask(result, h, w):
    mask = np.zeros((h,w),dtype=np.uint8)
    if result.masks is None: return mask
    best_s,best_m = -1,None
    for m in result.masks.data.cpu().numpy():
        m  = cv2.resize(m,(w,h),interpolation=cv2.INTER_LINEAR)
        bm = (m>MASK_THRESH).astype(np.uint8)
        sc = int(bm[int(h*.55):].sum())*2+int(bm.sum())
        if sc>best_s: best_s,best_m=sc,bm
    return best_m if best_m is not None else mask

def road_polygon(mask, w, h):
    def fallback():
        ty=int(h*HORIZON_FRAC)
        return np.array([(0,h-1),(int(w*.36),ty),(int(w*.64),ty),(w-1,h-1)],dtype=np.int32)
    if mask.sum()==0: return fallback()
    sm = cv2.GaussianBlur((mask*255).astype(np.uint8),(25,25),0)
    _,sm = cv2.threshold(sm,100,255,cv2.THRESH_BINARY)
    cnts,_ = cv2.findContours(sm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return fallback()
    lg = max(cnts,key=cv2.contourArea)
    if cv2.contourArea(lg)<w*h*0.04: return fallback()
    eps=0.01*cv2.arcLength(lg,True)
    ap=cv2.approxPolyDP(lg,eps,True).reshape(-1,2)
    return ap.astype(np.int32) if len(ap)>=4 else fallback()


# ══════════════════════════════════════════════════════════════════
# CANVAS DRAW HELPERS
# ══════════════════════════════════════════════════════════════════
def draw_lane_stripes(canvas, w, h, vp_x, vp_y):
    ego_h  = int(h*0.20)
    bot_y  = h-ego_h
    lx,rx  = int(w*0.02),int(w*0.98)
    rw     = rx-lx
    for s0,s1 in [(0.03,0.14),(0.18,0.32),(0.68,0.82),(0.86,0.97)]:
        bx1=lx+int(s0*rw); bx2=lx+int(s1*rw)
        tx1=int(vp_x+(bx1-vp_x)*0.06)
        tx2=int(vp_x+(bx2-vp_x)*0.06)
        pts=np.array([(bx1,bot_y),(bx2,bot_y),(tx2,vp_y),(tx1,vp_y)],dtype=np.int32)
        cv2.fillConvexPoly(canvas,pts,LANE_BGR)
        cv2.line(canvas,((bx1+bx2)//2,bot_y),((tx1+tx2)//2,vp_y),(255,255,255),1,cv2.LINE_AA)

def draw_ego_car(canvas, w, h):
    cx     = w//2
    ego_h  = int(h*0.20)
    base_y = h-4
    bw     = int(w*0.088)
    bh     = int(ego_h*0.82)
    bx1,by1 = cx-bw//2, base_y-bh
    bx2,by2 = cx+bw//2, base_y
    rw=int(bw*0.65); rh=int(bh*0.48)
    rx1,ry1=cx-rw//2,by1+int(bh*0.10)
    rx2,ry2=cx+rw//2,ry1+rh
    r=max(3,bw//9)
    body=np.array([[bx1+r,by1],[bx2-r,by1],[bx2,by1+r],[bx2,by2-r],
                   [bx2-r,by2],[bx1+r,by2],[bx1,by2-r],[bx1,by1+r]],dtype=np.int32)
    cv2.fillConvexPoly(canvas,body,(240,242,245))
    cv2.polylines(canvas,[body],True,(130,150,170),2,cv2.LINE_AA)
    cv2.rectangle(canvas,(rx1,ry1),(rx2,ry2),(195,215,230),-1)
    cv2.rectangle(canvas,(rx1,ry1),(rx2,ry2),(100,130,160),1)
    for pts in [
        np.array([[rx1+3,ry1],[rx2-3,ry1],[rx2-6,ry1+int(rh*.28)],[rx1+6,ry1+int(rh*.28)]],dtype=np.int32),
        np.array([[rx1+3,ry2],[rx2-3,ry2],[rx2-6,ry2-int(rh*.22)],[rx1+6,ry2-int(rh*.22)]],dtype=np.int32),
    ]:
        cv2.fillConvexPoly(canvas,pts,(140,195,230))
    hl_h=max(3,int(bh*0.06))
    for hx1,hx2 in [(bx1+4,bx1+int(bw*.30)),(bx2-int(bw*.30),bx2-4)]:
        cv2.rectangle(canvas,(hx1,by1+2),(hx2,by1+2+hl_h),(220,240,255),-1)
        cv2.rectangle(canvas,(hx1,by2-2-hl_h),(hx2,by2-2),(0,0,210),-1)
    ww=max(4,bw//8); wh=max(7,bh//6)
    for wx,wy in [(bx1-ww+2,by1+int(bh*.15)),(bx2-2,by1+int(bh*.15)),
                  (bx1-ww+2,by2-int(bh*.15)-wh),(bx2-2,by2-int(bh*.15)-wh)]:
        cv2.rectangle(canvas,(wx,wy),(wx+ww,wy+wh),(40,40,45),-1)
    cv2.line(canvas,(bx1+r,by2),(bx2-r,by2),EGO_ACCENT,2,cv2.LINE_AA)
    cv2.putText(canvas,"YOU",(cx-16,by2+16),cv2.FONT_HERSHEY_SIMPLEX,0.42,EGO_ACCENT,1,cv2.LINE_AA)

def dist_label_str(depth):
    if   depth < 0.25: return "!! VERY CLOSE"
    elif depth < 0.50: return "CLOSE"
    elif depth < 0.75: return "MEDIUM"
    else:              return "FAR"

def danger_bgr(depth):
    r = clamp(int((1-depth)*220),0,220)
    g = clamp(int(depth*180),   0,180)
    return (20, g, r)

def draw_label_below(canvas, cx, bottom_y, text, col):
    """Label di BAWAH model, bukan di atas."""
    fs=0.36; pad=3
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,fs,1)
    lx=cx-tw//2-pad; ly=bottom_y+4
    cv2.rectangle(canvas,(lx,ly),(lx+tw+pad*2,ly+th+pad*2),(18,18,22),-1)
    cv2.putText(canvas,text,(lx+pad,ly+th+1),cv2.FONT_HERSHEY_SIMPLEX,fs,col,1,cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    global _H_MAT

    # OpenGL context
    if not init_gl_context(OGL_W, OGL_H):
        sys.exit("Gagal init OpenGL context.")

    # 3D renderer
    if not os.path.exists(CAR_GLB_PATH):
        sys.exit(f"File GLB tidak ditemukan: {CAR_GLB_PATH}")
    renderer = CarRenderer(CAR_GLB_PATH)
    renderer.init_gl()
    print(f"[OK] GLB: {CAR_GLB_PATH}  ({len(renderer.mesh.vaos)} primitives)")

    # YOLO
    road_model    = YOLO(ROAD_YOLO_WEIGHTS)
    vehicle_model = YOLO(YOLO11N_WEIGHTS)

    # MiDaS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas  = torch.hub.load("intel-isl/MiDaS","MiDaS_small")
    tf_    = torch.hub.load("intel-isl/MiDaS","transforms").small_transform
    midas.to(device).eval()

    # Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): sys.exit(f"Tidak bisa buka: {VIDEO_PATH}")
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(OUTPUT_PATH,cv2.VideoWriter_fourcc(*"mp4v"),FPS,(W,H))

    # Build homography sekarang kita tahu W, H
    _H_MAT = build_homography(W, H)
    print(f"[OK] Video {W}×{H}@{FPS:.0f}fps  |  Homography ready")
    print("[>>] Processing... tekan Q untuk berhenti.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]

        # Road seg
        r_mask = get_road_mask(road_model(frame,conf=CONF_ROAD,verbose=False)[0],h,w)
        r_poly = road_polygon(r_mask,w,h)

        # Vehicle det
        v_res  = vehicle_model(frame,conf=CONF_VEHICLE,verbose=False)[0]
        dets   = []
        for box in v_res.boxes:
            cls = vehicle_model.names[int(box.cls[0])]
            if cls not in TARGET_CLASSES: continue
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            dets.append((x1,y1,x2,y2,float(box.conf[0]),cls))

        # MiDaS horizon
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            pred = midas(tf_(rgb).to(device))
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),size=rgb.shape[:2],mode="bicubic",align_corners=False).squeeze()
        dm  = cv2.normalize(pred.cpu().numpy(),None,0,1,cv2.NORM_MINMAX)
        cd  = float(dm[h//2,w//2])
        vp_y= clamp(int(h*(HORIZON_FRAC+0.05*(0.5-cd))),int(h*0.33),int(h*0.55))
        vp_x= w//2

        # ── Canvas ──────────────────────────────────────────────
        render = np.full_like(frame, SKY_BGR)
        cv2.fillPoly(render,[r_poly],ROAD_BGR)
        cv2.polylines(render,[r_poly],True,(185,185,190),2,cv2.LINE_AA)
        cv2.line(render,(0,vp_y),(w,vp_y),(55,55,60),1,cv2.LINE_AA)

        cs=np.array([(int(w*.44),h-1),(int(w*.51),h-1),(vp_x+4,vp_y),(vp_x-4,vp_y)],dtype=np.int32)
        cv2.fillConvexPoly(render,cs,(235,235,238))
        draw_lane_stripes(render,w,h,vp_x,vp_y)

        # ── Project & sort ──────────────────────────────────────
        proj_list = []
        for (x1,y1,x2,y2,conf,cls) in dets:
            bbox_h = max(1, y2-y1)
            bbox_w = max(1, x2-x1)
            aspect = clamp(bbox_w/bbox_h, 0.5, 3.5)

            # Warp bottom-centre of bbox to canvas
            foot_x = (x1+x2)//2
            foot_y = y2          # titik "kaki" mobil di jalan
            px, py = project_point(_H_MAT, foot_x, foot_y)

            # Ukuran model di canvas dari bbox height
            car_h = compute_canvas_size(bbox_h, h)
            car_w = int(car_h * aspect * 0.80)   # ×0.8: top-down tampak lebih sempit

            # Depth ratio untuk colour + sorting
            # Pakai kombinasi py (canvas y) sebagai depth indicator
            # Makin bawah di canvas = makin dekat
            ego_h     = int(h*0.20)
            road_top  = vp_y
            road_bot  = h - ego_h
            depth = 1.0 - clamp((py - road_top) / max(1, road_bot - road_top), 0.0, 1.0)

            proj_list.append((depth, x1,y1,x2,y2, conf, cls, px, py, car_w, car_h))

        # Far first → near last (near tampil di atas/depan)
        proj_list.sort(key=lambda t: -t[0])

        for (depth,x1,y1,x2,y2,conf,cls,px,py,cw,ch) in proj_list:
            col = danger_bgr(depth)

            # Render & composite 3D GLB
            renderer.composite_onto(render, px, py, cw, ch, depth)

            # Label di BAWAH model
            draw_label_below(render, px, py, f"{cls} {dist_label_str(depth)}", col)

            # Original frame annotation
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
            cv2.putText(frame,f"{cls} {conf:.2f} | {dist_label_str(depth)}",
                        (x1,max(16,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.46,col,2,cv2.LINE_AA)

        # Ego car selalu paling atas
        draw_ego_car(render,w,h)
        render = cv2.GaussianBlur(render,(3,3),0)

        cv2.imshow("Original Video",frame)
        cv2.imshow("Autonomous Driving View",render)
        writer.write(render)
        if cv2.waitKey(1)&0xFF==ord("q"): break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[DONE] → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()