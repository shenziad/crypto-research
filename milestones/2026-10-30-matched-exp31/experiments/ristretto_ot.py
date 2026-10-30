"""Local semi-honest research OT using Ristretto255 and SHA256/SHAKE256.

Chou-Orlandi 1-out-of-2 DH equations, independent ephemeral sender points;
then the standard logN seed-combination reduction for 1-out-of-N messages.
No new primitive, active/UC security, constant-time Python implementation,
OT extension, sublinear communication or deployment-hardening claim.
"""
import ctypes
import ctypes.util
import hashlib
import os
import struct


def xor(a, b):
    assert len(a) == len(b)
    return bytes(x ^ y for x, y in zip(a, b))


class Channel:
    def __init__(self, sock):
        self.sock = sock
        self.sent = self.received = self.sends = self.receives = 0

    def send(self, payload):
        packet = struct.pack('<Q', len(payload))+payload
        self.sock.sendall(packet)
        self.sent += len(packet)
        self.sends += 1

    def _read(self, n):
        data = bytearray()
        while len(data) < n:
            chunk = self.sock.recv(n-len(data))
            if not chunk:
                raise EOFError('peer closed')
            data.extend(chunk)
        return bytes(data)

    def recv(self):
        length = struct.unpack('<Q', self._read(8))[0]
        assert length < (1 << 31)
        payload = self._read(length)
        self.received += 8+length
        self.receives += 1
        return payload


class Group:
    def __init__(self):
        self.lib = ctypes.CDLL(ctypes.util.find_library('sodium'))
        assert self.lib.sodium_init() >= 0
        self.lib.sodium_version_string.restype = ctypes.c_char_p
        self.version = self.lib.sodium_version_string().decode()
        for name in ('crypto_scalarmult_ristretto255_base','crypto_scalarmult_ristretto255',
                     'crypto_core_ristretto255_add','crypto_core_ristretto255_sub',
                     'crypto_core_ristretto255_is_valid_point'):
            getattr(self.lib,name).restype = ctypes.c_int
        self.lib.crypto_core_ristretto255_scalar_random.restype = None

    @staticmethod
    def out():
        return ctypes.create_string_buffer(32)

    def scalar(self):
        while True:
            value = self.out()
            self.lib.crypto_core_ristretto255_scalar_random(value)
            if value.raw != bytes(32):
                return value.raw

    def valid(self, point):
        return len(point) == 32 and point != bytes(32) and self.lib.crypto_core_ristretto255_is_valid_point(point) == 1

    def base(self, scalar):
        out = self.out()
        assert len(scalar) == 32
        assert self.lib.crypto_scalarmult_ristretto255_base(out, scalar) == 0
        return out.raw

    def mul(self, scalar, point):
        assert len(scalar) == 32 and self.valid(point)
        out = self.out()
        assert self.lib.crypto_scalarmult_ristretto255(out,scalar,point) == 0
        return out.raw

    def add(self, left, right):
        out = self.out()
        assert self.lib.crypto_core_ristretto255_add(out,left,right) == 0
        return out.raw

    def sub(self, left, right):
        out = self.out()
        assert self.lib.crypto_core_ristretto255_sub(out,left,right) == 0
        return out.raw


def key(context, sid, index, sender, receiver, shared):
    return hashlib.sha256(b'CO-ristretto-research-v1\0'+struct.pack('<I',len(context))+context+
        sid+struct.pack('<I',index)+sender+receiver+shared).digest()


def ot2_send(channel, pairs, context=b'base'):
    assert pairs and all(len(a) == len(b) == 16 for a,b in pairs)
    group = Group(); sid = os.urandom(16)
    states = [(group.scalar(),) for _ in pairs]
    states = [(scalar,group.base(scalar)) for scalar, in states]
    channel.send(sid+b''.join(point for _,point in states))
    received = channel.recv()
    assert len(received) == 32*len(pairs)
    encrypted = []
    for i, ((scalar, point), (m0,m1)) in enumerate(zip(states,pairs)):
        response = received[i*32:(i+1)*32]
        difference = group.sub(response,point)
        assert group.valid(response) and group.valid(difference)
        k0 = key(context,sid,i,point,response,group.mul(scalar,response))
        k1 = key(context,sid,i,point,response,group.mul(scalar,difference))
        encrypted.extend((xor(m0,k0[:16]),xor(m1,k1[:16])))
    channel.send(b''.join(encrypted))


def ot2_receive(channel, choices, context=b'base'):
    assert choices and all(bit in (0,1) for bit in choices)
    group = Group(); first = channel.recv()
    assert len(first) == 16+32*len(choices)
    sid, senders = first[:16], first[16:]
    states = []; responses = []
    for i, bit in enumerate(choices):
        point = senders[i*32:(i+1)*32]
        assert group.valid(point)
        while True:
            scalar = group.scalar(); response = group.base(scalar)
            if bit: response = group.add(response,point)
            if group.valid(response) and group.valid(group.sub(response,point)):
                break
        states.append(key(context,sid,i,point,response,group.mul(scalar,point)))
        responses.append(response)
    channel.send(b''.join(responses))
    encrypted = channel.recv()
    assert len(encrypted) == 32*len(choices)
    return [xor(encrypted[32*i+16*bit:32*i+16*(bit+1)],states[i][:16]) for i,bit in enumerate(choices)]


def row_pad(context, nonce, index, seeds, width):
    return hashlib.shake_256(b'logN-OT-row-research-v1\0'+struct.pack('<I',len(context))+context+
        nonce+struct.pack('<Q',index)+b''.join(seeds)).digest(width)


def otn_send(channel, messages, context=b'array'):
    n = len(messages)
    assert n >= 2 and n & (n-1) == 0
    width = len(messages[0]); assert width > 0 and all(len(m) == width for m in messages)
    bits = n.bit_length()-1; nonce = os.urandom(16)
    channel.send(struct.pack('<QI',n,width)+nonce)
    seeds = [(os.urandom(16),os.urandom(16)) for _ in range(bits)]
    ot2_send(channel,seeds,context+nonce)
    encrypted = bytearray(n*width)
    for index,message in enumerate(messages):
        chosen = [seeds[i][(index>>i)&1] for i in range(bits)]
        encrypted[index*width:(index+1)*width] = xor(message,row_pad(context,nonce,index,chosen,width))
    channel.send(bytes(encrypted))


def otn_receive(channel, index, n, width, context=b'array'):
    assert 0 <= index < n and n >= 2 and n & (n-1) == 0
    first = channel.recv(); assert len(first) == 28
    expected_n,expected_width = struct.unpack('<QI',first[:12])
    assert expected_n == n and expected_width == width
    nonce = first[12:]; bits=n.bit_length()-1
    chosen=ot2_receive(channel,[(index>>i)&1 for i in range(bits)],context+nonce)
    encrypted=channel.recv(); assert len(encrypted) == n*width
    return xor(encrypted[index*width:(index+1)*width],row_pad(context,nonce,index,chosen,width))
