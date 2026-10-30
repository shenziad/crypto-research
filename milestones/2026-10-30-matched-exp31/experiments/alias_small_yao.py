"""Unoptimized, independent-label four-row Yao circuit for the alias finish.

Research implementation in a passive random-oracle transcript model. No new
garbling, free-XOR/half-gates, active security or side-channel claim. Receiver
input labels use actual Ristretto OT; there is no trusted evaluator/helper.
"""
import hashlib
import os
import struct

from ristretto_ot import ot2_send, ot2_receive, xor


def bits(value, width):
    assert 0 <= value < (1 << width)
    return [(value>>i)&1 for i in range(width)]


class Circuit:
    def __init__(self, index_bits, precision):
        ell,p=index_bits,precision
        self.sender_count=3*ell+2*p+1
        self.receiver_count=2*ell+2*p+1
        self.next_wire=1+self.sender_count+self.receiver_count
        self.gates=[]
        sender=[(i,0) for i in range(1,1+self.sender_count)]
        receiver=[(i,0) for i in range(1+self.sender_count,self.next_wire)]
        def split(values,widths):
            rows=[];start=0
            for width in widths:rows.append(values[start:start+width]);start+=width
            assert start==len(values)
            return rows
        kappa,cutmask,aliasmask,us,rho=split(sender,[ell,p+1,ell,p,ell])
        j,cutshare,aliasshare,ur=split(receiver,[ell,p+1,ell,p])
        column=[self.gate('xor',a,b) for a,b in zip(kappa,j)]
        threshold=[self.gate('xor',a,b) for a,b in zip(cutmask,cutshare)]
        alternate=[self.gate('xor',a,b) for a,b in zip(aliasmask,aliasshare)]
        uniform=[self.gate('xor',a,b) for a,b in zip(us,ur)]+[(0,0)]
        borrow=(0,0)
        # Borrow of U-threshold is majority(not U_i, threshold_i, borrow).
        # majority(a,b,c) = a XOR ((a XOR b) AND (a XOR c)).
        for u,t in zip(uniform,threshold):
            a=(u[0],u[1]^1)
            borrow=self.gate('xor',a,self.gate('and',self.gate('xor',a,t),self.gate('xor',a,borrow)))
        selected=[self.gate('xor',a,self.gate('and',borrow,self.gate('xor',a,c)))
                  for a,c in zip(alternate,column)]
        self.outputs=[self.gate('xor',s,r) for s,r in zip(selected,rho)]

    def gate(self,kind,left,right):
        out=self.next_wire;self.next_wire+=1
        self.gates.append((kind,left,right,out))
        return out,0

    def plain(self,sender,receiver):
        assert len(sender)==self.sender_count and len(receiver)==self.receiver_count
        values=[0]+sender+receiver
        for kind,left,right,out in self.gates:
            a=values[left[0]]^left[1];b=values[right[0]]^right[1]
            assert len(values)==out
            values.append(a^b if kind=='xor' else a&b)
        return sum((values[wire]^invert)<<i for i,(wire,invert) in enumerate(self.outputs))


def labels():
    first=os.urandom(16);second=bytearray(os.urandom(16))
    second[-1]=(second[-1]&254)|((first[-1]&1)^1)
    return first,bytes(second)


def gate_pad(sid,number,left,right,a,b):
    context=struct.pack('<III BB',number,left[0],right[0],left[1],right[1])
    return hashlib.sha256(b'four-row-Yao-research-v1\0'+sid+context+a+b).digest()[:16]


def send(channel,circuit,sender):
    assert len(sender)==circuit.sender_count
    sid=os.urandom(16);wires=[labels() for _ in range(circuit.next_wire)]
    encoded=[sid,wires[0][0]]
    encoded += [wires[i+1][bit] for i,bit in enumerate(sender)]
    for number,(kind,left,right,out) in enumerate(circuit.gates):
        rows=[None]*4
        for av in (0,1):
            for bv in (0,1):
                a,b=wires[left[0]][av],wires[right[0]][bv]
                x,y=av^left[1],bv^right[1]
                result=x^y if kind=='xor' else x&y
                row=((a[-1]&1)<<1)|(b[-1]&1)
                rows[row]=xor(wires[out][result],gate_pad(sid,number,left,right,a,b))
        encoded.extend(rows)
    encoded.append(bytes((wires[wire][0][-1]&1)^invert for wire,invert in circuit.outputs))
    channel.send(b''.join(encoded))
    start=1+circuit.sender_count
    ot2_send(channel,wires[start:start+circuit.receiver_count],b'alias-Yao-inputs')


def receive(channel,circuit,receiver):
    assert len(receiver)==circuit.receiver_count
    payload=channel.recv()
    count=len(circuit.gates)
    assert len(payload)==16+16*(1+circuit.sender_count)+64*count+len(circuit.outputs)
    sid=payload[:16];at=16
    wires=[]
    for _ in range(1+circuit.sender_count):wires.append(payload[at:at+16]);at+=16
    wires+=ot2_receive(channel,receiver,b'alias-Yao-inputs')
    for number,(_,left,right,out) in enumerate(circuit.gates):
        a,b=wires[left[0]],wires[right[0]]
        row=((a[-1]&1)<<1)|(b[-1]&1)
        ciphertext=payload[at+16*row:at+16*(row+1)]
        assert len(wires)==out
        wires.append(xor(ciphertext,gate_pad(sid,number,left,right,a,b)))
        at+=64
    decoding=payload[at:]
    return sum(((wires[wire][-1]&1)^decoding[i])<<i for i,(wire,_) in enumerate(circuit.outputs))


def sender_bits(kappa,cutmask,aliasmask,us,rho,ell,p):
    return bits(kappa,ell)+bits(cutmask,p+1)+bits(aliasmask,ell)+bits(us,p)+bits(rho,ell)


def receiver_bits(j,cutshare,aliasshare,ur,ell,p):
    return bits(j,ell)+bits(cutshare,p+1)+bits(aliasshare,ell)+bits(ur,p)
