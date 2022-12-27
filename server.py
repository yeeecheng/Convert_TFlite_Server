import socket
import threading
import os
import subprocess
import shutil
import yaml
HOST  = "127.0.0.1"
PORT = 80


active_thread=[0]*5
# 接受資料
def receive_service(conn, addr,thread_num):
    
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
            
            receive_data(conn,thread_num)
            
        # 接受yaml
        elif mode == "yaml":
            
            receive_data(conn,thread_num)
            
        # # 結束
        
        
    print("receive data finish")

# 處理接收的資料
def receive_data(conn,thread_num):
    
    info = conn.recv(1024)
    size , file_path = info.decode().split('|')
    
    if size and file_path:
        
       
        new_file = open(os.path.join(f"dataset_{thread_num}",file_path),"wb")
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
def send_tflite(conn):
    
    path = "./yolov5/runs/train/exp/weights/best-int8_edgetpu.tflite"
    conn.send(b"tflite")
    reply = conn.recv(1024)
    
    if reply.decode() == "get mode":
        
        file = open(path,"rb")
        file_bytes = file.read()
        
        conn.send("{}|{}".format(len(file_bytes),path).encode())
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
def update_yaml(thread_num):
    
    with open(os.path.join(f"./dataset_{thread_num}","./dataset.yaml"),'r',encoding= 'utf-8') as f:
        data = yaml.load(f,Loader=yaml.FullLoader)
    
    train_path = data["train"]
    val_path = data['val'] 
    data["train"] =os.path.join(f"./dataset_{thread_num}",train_path)
    data['val']  =os.path.join(f"./dataset_{thread_num}",val_path)
   
    # 寫檔
    with open(os.path.join(f"./dataset_{thread_num}","./dataset.yaml"),'w',encoding= 'utf-8') as f:
        yaml.dump(data,f)
        
def run(connect,addr,thread_num):
    
    
   
    #  # 接收資料
    receive_service(connect,addr,thread_num)
    
    # 刪除原本的訓練檔
    directory_path ="./yolov5/runs/train/exp"
    if os.path.isdir(directory_path):
        try:
            shutil.rmtree(directory_path)
        except OSError as e:
            print(e)
            connect.send(b"server Error")
            connect.close()
        else:
            print("Delete successfully")
    update_yaml(thread_num)
    with open(os.path.join(f"./dataset_{thread_num}","./dataset.yaml"),'r',encoding= 'utf-8') as f:
        data = yaml.load(f,Loader=yaml.FullLoader)

    # 執行 yolov5 train.py 訓練辨識
    args = f"python3 ./yolov5/train.py --img 320 --batch 16 --epoch 1 --data ./dataset_{thread_num}/dataset.yaml --weight yolov5s.pt --cfg ./yolov5/models/yolov5s.yaml".split(' ')
    ret = subprocess.run(args)

    
    # 執行 yolov5 export.py 轉成edgetpu.tflite
    args = f"python3 ./yolov5/export.py --weight ./yolov5/runs/train/exp/weights/best.pt --data ./dataset_{thread_num}/dataset.yaml --img 320 --include edgetpu".split(" ")
    ret = subprocess.run(args)
    
            
    # 傳送edgetpu.tflite
    print("sent tflite")
    send_tflite(connect)
    connect.send(b'end')
    shutil.rmtree(f"./dataset_{thread_num}")
    active_thread[thread_num]=0
    connect.close()

def create_server():
    
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
        print(addr)
        if cur_cnt == 5:
            connect.close()
                
        for i,res in enumerate(active_thread):
            if not res:
               
                thread_num = i
                active_thread[i]=1
                break
        print(thread_num)
        if os.path.isdir(f"./dataset_{thread_num}"):
            shutil.rmtree(f"./dataset_{thread_num}")
        os.mkdir(f"./dataset_{thread_num}")
        os.mkdir(f"./dataset_{thread_num}/dataset")
        os.mkdir(f"./dataset_{thread_num}/dataset/images")
        os.mkdir(f"./dataset_{thread_num}/dataset/labels")
        os.mkdir(f"./dataset_{thread_num}/dataset/images/train")
        os.mkdir(f"./dataset_{thread_num}/dataset/images/val")
        os.mkdir(f"./dataset_{thread_num}/dataset/labels/train")
        os.mkdir(f"./dataset_{thread_num}/dataset/labels/val")
        print("!")
        t = threading.Thread(target=run ,args=(connect,addr,thread_num))
        t.start()
       

    
if __name__ == "__main__":
    create_server()