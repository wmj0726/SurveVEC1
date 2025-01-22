import asyncio
import hashlib
import random
import time
from typing import List

import numpy as np
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature
from concurrent.futures import ThreadPoolExecutor


class Block:
    def __init__(self, index: int, previous_hashes: List[str], timestamp: float, data: str, hash: str,
                 signature: bytes):
        self.index = index
        self.previous_hashes = previous_hashes  # List of previous block hashes
        self.timestamp = timestamp
        self.data = data
        self.hash = hash
        self.signature = signature

    def __repr__(self):
        return f"Block(index={self.index}, hash={self.hash}, previous_hashes={self.previous_hashes}, timestamp={self.timestamp}, data={self.data}, signature={self.signature.hex()})"


class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, ["0"], time.time(), "Genesis Block",
                              self.calculate_hash(0, ["0"], time.time(), "Genesis Block"), b"")
        self.chain.append(genesis_block)

    def get_last_blocks(self):
        return [self.chain[-1]]

    def calculate_hash(self, index: int, previous_hashes: List[str], timestamp: float, data: str):
        value = f"{index}{''.join(previous_hashes)}{timestamp}{data}"
        return hashlib.sha256(value.encode()).hexdigest()

    def add_block_dag(self, data: str, node_id: int, private_key):
        last_blocks = self.get_last_blocks()
        index = last_blocks[0].index + 1
        previous_hashes = [block.hash for block in last_blocks]
        timestamp = time.time()
        hash = self.calculate_hash(index, previous_hashes, timestamp, data)
        signature = self.sign_data(private_key, data)
        new_block = Block(index, previous_hashes, timestamp, data, hash, signature)
        if self.verify_block(new_block, private_key.public_key()):
            self.chain.append(new_block)
            #print(f"Node {node_id} added Block {new_block}")  # Print statement to track block addition
        else:
            print(f"Node {node_id} failed to add Block {new_block} due to invalid signature")

    def add_block_pow(self, data: str, node_id: int, private_key, publicKey):
        last_blocks = self.get_last_blocks()
        index = last_blocks[0].index + 1
        previous_hashes = [block.hash for block in last_blocks]
        timestamp = time.time()
        nonce, hash = self.proof_of_work(index, previous_hashes, timestamp, data)
        print(private_key.public_key)
        signature = self.sign_data(private_key, data)
        new_block = Block(index, previous_hashes, timestamp, data, hash, signature)
        if self.verify_block(new_block, private_key.public_key()):
            self.chain.append(new_block)
            #print(f"Node {node_id} added Block {new_block}")
        else:
            print(f"Node {node_id} failed to add Block {new_block} due to invalid signature")

    def proof_of_work(self, index: int, previous_hashes: List[str], timestamp: float, data: str, difficulty=2):
        nonce = 0
        hash = self.calculate_hash(index, previous_hashes, timestamp, data)
        while not hash.startswith('0' * difficulty):
            nonce += 1
            hash = self.calculate_hash(index, previous_hashes, timestamp, f"{data}{nonce}")
        return nonce, hash



    def sign_data(self, private_key, data):
        return private_key.sign(
            str(data).encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

    def verify_signature(self, public_key, data, signature):
        try:
            public_key.verify(
                signature,
                str(data).encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False

    def verify_block(self, block, public_key):
        return self.verify_signature(public_key, block.data, block.signature)


class Node:
    def __init__(self, node_id: int, blockchain: Blockchain, stake: int, data, private_key, public_key):
        self.node_id = node_id
        self.blockchain = blockchain
        self.stake = stake
        self.data = data
        self.private_key = private_key
        self.public_key = public_key

    def mine_block_dag(self):
        data = self.data
        self.blockchain.add_block_dag(data, self.node_id, self.private_key)


def select_node(nodes: List[Node]):
    total_stake = sum(node.stake for node in nodes)
    pick = random.uniform(0, total_stake)
    current = 0
    for node in nodes:
        current += node.stake
        if current > pick:
            return node

def simulate_pow(blockchain: Blockchain, car_msg, private_key):
    node_id = 2  # Simulate a single node mining
    blockchain.add_block_pow(car_msg, node_id, private_key)
    return blockchain

async def simulate_hybrid(blockchain: Blockchain, nodes: List[Node], car_msg, consense_type):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        tasks = []
        for msg in car_msg:
            selected_node = select_node(nodes)
            selected_node.data = msg  # Update the data to be mined
            task = loop.run_in_executor(executor, selected_node.mine_block_dag)
            tasks.append(task)
        await asyncio.gather(*tasks)

def generate_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    return private_key, public_key


async def main(blockchain, hybrid_nodes, msg, consense_type):
    # Simulate hybrid consensus with DAG structure
    await simulate_hybrid(blockchain, hybrid_nodes, msg, consense_type)

def asynci(blockchain, hybrid_nodes, msg, consense_type):
    asyncio.run(main(blockchain, hybrid_nodes, msg, consense_type))

if __name__=='main':
    blockchan = Blockchain()
    private_key, public_key = generate_keys()
