import socket
import pickle


class SocketConnector:
    def __init__(self, port=1234):
        self.port    = port

    def connectAsClient(self, host, n_timeout=10):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.sock.setblocking(0)
        print('waiting for connections as Client of {}:{}...'\
                .format(host, self.port))
        while True:
            try:
                self.sock.connect((host, self.port))
            except:
                continue
            break
        self.sock.settimeout(n_timeout)
        print('connecion is done')

    def connectAsServer(self, n_que=1, n_timeout=10):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('', self.port))
        server.listen(n_que)

        print('waiting for connections as Server of localhost:{}...'\
                .format(self.port))
        self.sock, client_address = server.accept()
        self.sock.settimeout(n_timeout)
        #self.sock.setblocking(0)
        print('connecion is done')

    def sendData(self, obj, buf_size=2**10):
        byte = pickle.dumps(obj)
        self.sock.sendall(byte)
        self.sock.sendall(b'1234321')

    def recvData(self, buf_size=2**10):
        data = b''
        while True:
            try:
                packet = self.sock.recv(buf_size)
                data  += packet
                #print('{}:{}'.format(data,packet))
                if packet[-7:] == b'1234321' and len(data) != 0:
                    #print("data is {}".format(data))
                    break
            except socket.timeout:
                break
            except KeyboardInterrupt:
                break
            except:
                continue

            #packet  = self.sock.recv(buf_size)
            #data = packet
            #break
            #data   += packet
            #if len(packet) == 0 and len(data) != 0:
            #    print("data is {}".format(data))
            #    break
            #try:
            #    packet  = self.sock.recv(buf_size)
            #    data   += packet
            #    if len(packet) == 0 and len(data) != 0:
            #        print("data is {}".format(data))
            #        break
            #except socket.timeout:
            #    break
            #except BlockingIOError:
            #    continue
            #except:
            #    continue

        #self.sock.send(b'')
        #print(data)
        return pickle.loads(data)

    def close(self,):
        self.sock.shutdown(1)
        self.sock.close()
        print('connection closed')
