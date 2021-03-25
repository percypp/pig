# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:04:24 2021

@author: gordon
"""

# Python program to implement client side of chat room. 
import socket 
import select 
import time
import sys 


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
if len(sys.argv) != 3: 
    print ("Correct usage: script, IP address, port number") 
    exit() 
IP_address = str(sys.argv[1]) 
Port = int(sys.argv[2]) 
server.connect((IP_address, Port)) 
todo=""
mode="none"
while True: 

    # maintains a list of possible input streams 
    sockets_list = [ server] #sys.stdin,

    """ There are two possible input situations. Either the 
    user wants to give manual input to send to other people, 
    or the server is sending a message to be printed on the 
    screen. Select returns from sockets_list, the stream that 
    is reader for input. So for example, if the server wants 
    to send a message, then the if condition will hold true 
    below.If the user wants to send a message, the else 
    condition will evaluate as true"""
    read_sockets,write_socket, error_socket = select.select(sockets_list,[],[],0.1) 
    for socks in read_sockets:
        if socks == server:
            message = socks.recv(2048)
            commend=message.decode()
            commend=commend.split()
            print (commend)
            if(commend[0]=="mode1"):
                todo="color-depth"
            elif(commend[0]=="mode2"):
                todo="color*6"
            elif(commend[0]=="mode3"):
                todo="depth*6"
            elif(commend[0]=="rec"):
                if(len(commend)==3):
                    todo="record :"+commend[1]+".bag"
                    print(todo)
                    time.sleep(int(commend[2]))
            elif(commend[0]=="set"):
                if(len(commend)==2 and(commend[1]=="see" or commend[1]=="review")):
                    mode=commend[1]
            else :
                mode="none"
                todo=""
        else: 
            message = sys.stdin.readline() 
            server.send(message)
            sys.stdout.write("<You>") 
            sys.stdout.write(message) 
            sys.stdout.flush() 
    print(mode+" "+todo)
    if(mode!="none"):
        server.send((mode+" "+todo).encode())
server.close() 
