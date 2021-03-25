# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:03:42 2021

@author: gordon
python ./server2.py 127.0.0.1 4567
"""
# Python program to implement server side of chat room. 
import socket 
import select 
import sys 
from _thread import *
import tkinter as tk        # python v3
import os

"""The first argument AF_INET is the address domain of the 
socket. This is used when we have an Internet Domain with 
any two hosts The second argument is the type of socket. 
SOCK_STREAM means that data or characters are read in 
a continuous flow."""
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 

# checks whether sufficient arguments have been provided 
if len(sys.argv) != 3: 
	print ("Correct usage: script, IP address, port number") 
	exit() 

# takes the first argument from command prompt as IP address 
IP_address = str(sys.argv[1]) 

# takes second argument from command prompt as port number 
Port = int(sys.argv[2]) 

""" 
binds the server to an entered IP address and at the 
specified port number. 
The client must be aware of these parameters 
"""
server.bind((IP_address, Port)) 

""" 
listens for 100 active connections. This number can be 
increased as per convenience. 
"""
server.listen(100) 

list_of_clients = [] 
def callback(input_entry):
    print("User entered : " + input_entry)
    broadcast(input_entry.encode(),None)
   # os.system('touch '+str(input_entry.get())+'.txt')
    return None

def testinput():
    
    root = tk.Tk()
    root.geometry('550x550')   #Set window size

# Heading
    heading = tk.Label(root, text="A simple GUI")
    heading.place(x = 100, y = 0)
    input_label = tk.Label(root, text="filename(space)record time")
    input_label.place(x = 0, y = 190)
    input_entry = tk.Entry(root)
    input_entry.place(x = 200, y = 190)
    input_entry2 = tk.Entry(root)
    input_entry2.place(x = 400, y = 190)
    submit_button = tk.Button(root, text = "watch review", command = lambda: callback("set watch"))
    submit_button.place(x = 200, y = 90)
    see_button = tk.Button(root, text = "camera viewer", command = lambda: callback("set see"))
    see_button.place(x = 200, y = 140)
    see_button = tk.Button(root, text = "record", command = lambda: callback("rec "+input_entry.get()))
    see_button.place(x = 200, y = 240)
    root.mainloop()
    """
    while(True):
        a=raw_input()
        print(a)
        broadcast(a.encode(),None)"""
        
def clientthread(conn, addr): 

	# sends a message to the client whose user object is conn 
	conn.send("Welcome to this chatroom!".encode()) 

	while True:
			#print("in in in")
			try: 
				message = conn.recv(2048) 
				if message: 

					"""prints the message and address of the 
					user who just sent the message on the server 
					terminal"""
					print ("<" + addr[0] + "> " + message.decode()) 

					# Calls broadcast function to send message to all 
					message_to_send = "<" + addr[0] + "> " + message.decode()
					#broadcast(message_to_send.encode(), conn) 

				else: 
					"""message may have no content if the connection 
					is broken, in this case we remove the connection"""
					remove(conn) 

			except Exception as e: 
				print(e)
				continue

"""Using the below function, we broadcast the message to all 
clients who's object is not the same as the one sending 
the message """
def broadcast(message, connection): 
	for clients in list_of_clients: 
		if clients!=connection: 
			try: 
				clients.send(message) 
			except: 
				clients.close() 

				# if the link is broken, we remove the client 
				remove(clients) 

"""The following function simply removes the object 
from the list that was created at the beginning of 
the program"""
def remove(connection): 
	if connection in list_of_clients: 
		list_of_clients.remove(connection) 
start_new_thread(testinput,())
while True: 

	"""Accepts a connection request and stores two parameters, 
	conn which is a socket object for that user, and addr 
	which contains the IP address of the client that just 
	connected"""
	conn, addr = server.accept() 

	"""Maintains a list of clients for ease of broadcasting 
	a message to all available people in the chatroom"""
	list_of_clients.append(conn) 

	# prints the address of the user that just connected 
	print (addr[0] + " connected") 

	# creates and individual thread for every user 
	# that connects 
	start_new_thread(clientthread,(conn,addr)) 

conn.close() 
server.close() 

