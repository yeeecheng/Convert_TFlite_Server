import socket
import threading
import os
import subprocess
import shutil
import yaml
import argparse


active_thread=[0]*5
# 接受資料
def receive_service(conn, addr,thread_num,ROOT):
    
    while True:
       # 確認模式 ,接收什麼資料 
        mode = conn.recv(1024).decode()
        
        if mode == "end":
            print("end")
            break
        # check 
        conn.send(b"get mode")
        
        # 接受dataset
        if mode == "dataset":
            
            receive_data(conn,thread_num,ROOT)
            
        # 接受yaml
        elif mode == "yaml":
            
            receive_data(conn,thread_num,ROOT)
            
        # # 結束
        
        
    print("receive data finish")

# 處理接收的資料
def receive_data(conn,thread_num,ROOT):
    
    info = conn.recv(1024)
    size , file_path = info.decode().split('|')
    
    if size and file_path:
        
       
        new_file = open(os.path.join(ROOT,f"./dataset_{thread_num}",file_path),"wb")
        conn.send(b'ok')
        
        file = b''
        total_size =int(size)

        get_size = 0
        while get_size < total_size:

            data =conn.recv(1024)
            file+=data
            get_size +=len(data)
            
        new_file.write(file[:])
        new_file.close()
        conn.send(b'success')

# 回傳edgetpu_tflite
def send_tflite(conn,ROOT):
    
    tflite_path ="./yolov5/runs/train/exp/weights/best-int8_edgetpu.tflite"
    path = os.path.join(ROOT,tflite_path)
    conn.send(b"tflite")
    reply = conn.recv(1024)
    
    if reply.decode() == "get mode":
        
        file = open(path,"rb")
        file_bytes = file.read()
        
        conn.send("{}|{}".format(len(file_bytes),tflite_path).encode())
        reply = conn.recv(1024)
        
        if reply.decode() =="ok":
            total_size = len(file_bytes)
            cur = 0 
            while cur <total_size:
                data = file_bytes[cur:cur+1024]
                conn.send(data)
                cur+=len(data)
        
        reply = conn.recv(1024)
        if reply.decode() == "success":
            print("success")
        file.close()


# 更新yaml
def update_yaml(thread_num,ROOT):
    yaml_path = os.path.join(ROOT,f"./dataset_{thread_num}")
    with open(os.path.join(yaml_path,"./dataset.yaml"),'r',encoding= 'utf-8') as f:
        data = yaml.load(f,Loader=yaml.FullLoader)
    
    train_path = data["train"]
    val_path = data['val'] 
    data["train"] =os.path.join(yaml_path,train_path)
    data['val']  =os.path.join(yaml_path,val_path)
   
    # 寫檔
    with open(os.path.join(yaml_path,"./dataset.yaml"),'w',encoding= 'utf-8') as f:
        yaml.dump(data,f)
        
def run_thread(connect,addr,thread_num,ROOT):
    
    
   
    #  # 接收資料
    receive_service(connect,addr,thread_num,ROOT)
    
    # 刪除原本的訓練檔
    directory_path =os.path.join(ROOT,"./yolov5/runs/train/exp")
    if os.path.isdir(directory_path):
        try:
            shutil.rmtree(directory_path)
        except OSError as e:
            print(e)
            connect.send(b"server Error")
            connect.close()
        else:
            print("Delete successfully")
    update_yaml(thread_num,ROOT)
    with open(os.path.join(ROOT,f"./dataset_{thread_num}","./dataset.yaml"),'r',encoding= 'utf-8') as f:
        data = yaml.load(f,Loader=yaml.FullLoader)

    # 執行 yolov5 train.py 訓練辨識
    train_file_path = os.path.join(ROOT,"./yolov5/train.py")
    yaml_path = os.path.join(ROOT,f"./dataset_{thread_num}/dataset.yaml")
    weight_path = os.path.join(ROOT,"yolov5s.pt")
    cfg_path =os.path.join(ROOT,"./yolov5/models/yolov5s.yaml")
    args = f"python3 {train_file_path} --img 320 --batch 16 --epoch 1 --data {yaml_path} --weight {weight_path} --cfg {cfg_path}".split(' ')
    ret = subprocess.run(args)

    export_file_path = os.path.join(ROOT,"./yolov5/export.py")
    yaml_path = os.path.join(ROOT,f"./dataset_{thread_num}/dataset.yaml")
    weight_path = os.path.join(ROOT,"./yolov5/runs/train/exp/weights/best.pt")

    # 執行 yolov5 export.py 轉成edgetpu.tflite
    args = f"python3 {export_file_path} --weight {weight_path} --data {yaml_path} --img 320 --include edgetpu".split(" ")
    ret = subprocess.run(args)
    
            
    # 傳送edgetpu.tflite
    print("sent tflite")
    send_tflite(connect,ROOT)
    connect.send(b'end')
    shutil.rmtree(os.path.join(ROOT,f"./dataset_{thread_num}"))
    active_thread[thread_num]=0
    connect.close()

def create_server(opt):
    
    HOST  = opt.ip
    PORT = opt.port
    ROOT = opt.root
    #建立server ,等待連線
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind((HOST,PORT))
    s.listen(5)

    
    
    print(f"server start at {HOST} {PORT}")
    print("wait connect")
    cur_cnt=0

    while True:
        
        # client連上
        connect , addr = s.accept()
        print("connect!")
        if cur_cnt == 5:
            connect.close()
                
        for i,res in enumerate(active_thread):
            if not res:
               
                thread_num = i
                active_thread[i]=1
                break
        print(f"connect to the thread {thread_num}")
        if os.path.isdir(os.path.join(ROOT,f"./dataset_{thread_num}")):
            shutil.rmtree(os.path.join(ROOT,f"./dataset_{thread_num}"))
        os.mkdir(os.path.join(ROOT,f"./dataset_{thread_num}"))
        os.mkdir(os.path.join(ROOT,f"./dataset_{thread_num}/dataset"))
        os.mkdir(os.path.join(ROOT,f"./dataset_{thread_num}/dataset/images"))
        os.mkdir(os.path.join(ROOT,f"./dataset_{thread_num}/dataset/labels"))
        os.mkdir(os.path.join(ROOT,f"./dataset_{thread_num}/dataset/images/train"))
        os.mkdir(os.path.join(ROOT,f"./dataset_{thread_num}/dataset/images/val"))
        os.mkdir(os.path.join(ROOT,f"./dataset_{thread_num}/dataset/labels/train"))
        os.mkdir(os.path.join(ROOT,f"./dataset_{thread_num}/dataset/labels/val"))

        t = threading.Thread(target=run_thread ,args=(connect,addr,thread_num,ROOT))
        t.start()
       
    
def main(opt):
    create_server(opt)
    
def parse_opt(known=False):
   
    ROOT =os.getcwd()
    parser =argparse.ArgumentParser()
    parser.add_argument("--root",type=str , default=ROOT )
    parser.add_argument("--ip",type=str ,required=True) 
    parser.add_argument("--port",type=int,required=True)
  
    return parser.parse_args()[0] if known else parser.parse_args()


def run(**kwargs):

    opt = parse_opt(True)
    for k,v in kwargs.items():
        setattr(opt,k,v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)