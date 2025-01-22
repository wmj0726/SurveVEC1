import asyncio
import hashlib
import random
import time
import matplotlib.pyplot as plt
from typing import List

import numpy as np
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature
from concurrent.futures import ThreadPoolExecutor

class Block:
    def __init__(self, index: int, previous_hashes: List[str], timestamp: float, data: str, hash: str, signature: bytes):
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
        genesis_block = Block(0, ["0"], time.time(), "Genesis Block", self.calculate_hash(0, ["0"], time.time(), "Genesis Block"), b"")
        self.chain.append(genesis_block)

    def get_last_blocks(self):
        return [self.chain[-1]]

    def calculate_hash(self, index: int, previous_hashes: List[str], timestamp: float, data: str):
        value = f"{index}{''.join(previous_hashes)}{timestamp}{data}"
        return hashlib.sha256(value.encode()).hexdigest()

    def add_block_pow(self, data: str, node_id: int, private_key):
        last_blocks = self.get_last_blocks()
        index = last_blocks[0].index + 1
        previous_hashes = [block.hash for block in last_blocks]
        timestamp = time.time()
        nonce, hash = self.proof_of_work(index, previous_hashes, timestamp, data)
        signature = self.sign_data(private_key, data)
        new_block = Block(index, previous_hashes, timestamp, data, hash, signature)
        if self.verify_block(new_block, private_key.public_key()):
            self.chain.append(new_block)
            #print(f"Node {node_id} added Block {new_block}")
        else:
            print(f"Node {node_id} failed to add Block {new_block} due to invalid signature")

    def proof_of_work(self, index: int, previous_hashes: List[str], timestamp: float, data: str, difficulty=3):
        nonce = 0
        hash = self.calculate_hash(index, previous_hashes, timestamp, data)
        while not hash.startswith('0' * difficulty):
            nonce += 1
            hash = self.calculate_hash(index, previous_hashes, timestamp, f"{data}{nonce}")
        return nonce, hash

    def add_block_pos(self, data: str, node_id: int, private_key):
        last_blocks = self.get_last_blocks()
        index = last_blocks[0].index + 1
        previous_hashes = [block.hash for block in last_blocks]
        timestamp = time.time()
        hash = self.calculate_hash(index, previous_hashes, timestamp, data)
        signature = self.sign_data(private_key, data)
        new_block = Block(index, previous_hashes, timestamp, data, hash, signature)
        if self.verify_block(new_block, private_key.public_key()):
            self.chain.append(new_block)
            #print(f"Node {node_id} added Block {new_block}")
        else:
            print(f"Node {node_id} failed to add Block {new_block} due to invalid signature")

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
            #print(f"Node {node_id} added Block {new_block}")
        else:
            print(f"Node {node_id} failed to add Block {new_block} due to invalid signature")

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

    def tamper_chain(self):
        if len(self.chain) < 2:
            return False
        block_to_tamper = random.choice(self.chain[1:])  # Do not tamper with the genesis block
        original_data = block_to_tamper.data
        block_to_tamper.data = np.random.choice(100000)
        block_to_tamper.hash = self.calculate_hash(block_to_tamper.index, block_to_tamper.previous_hashes, block_to_tamper.timestamp, block_to_tamper.data)
        for block in self.chain[block_to_tamper.index + 1:]:
            block.previous_hashes[0] = block_to_tamper.hash
            block.hash = self.calculate_hash(block.index, block.previous_hashes, block.timestamp, block.data)
        return block_to_tamper.data != original_data

class Node:
    def __init__(self, node_id: int, blockchain: Blockchain, stake: int, data, private_key, public_key):
        self.node_id = node_id
        self.blockchain = blockchain
        self.stake = stake
        self.private_key = private_key
        self.public_key = public_key
        self.data = data

    def mine_block_pos(self):
        data = self.data
        self.blockchain.add_block_pos(data, self.node_id, self.private_key)

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
    local_blockchain = Blockchain()
    for i in range(len(car_msg)):
        node_id = i  # Simulate a single node mining
        local_blockchain.add_block_pow(car_msg[node_id], node_id, private_key)
    return local_blockchain

def simulate_pos(blockchain: Blockchain, nodes: List[Node], car_msg):
    local_blockchain = Blockchain()
    local_nodes = [Node(node.node_id, local_blockchain, node.stake, node.data, node.private_key, node.public_key) for node in nodes]
    for i in range(len(car_msg)):
        selected_node = select_node(local_nodes)
        selected_node.mine_block_pos()
    return local_blockchain

async def simulate_hybrid(blockchain: Blockchain, nodes: List[Node], car_msg):
    loop = asyncio.get_event_loop()
    local_blockchain = Blockchain()
    local_nodes = [Node(node.node_id, local_blockchain, node.stake, node.data, node.private_key, node.public_key) for node in nodes]
    with ThreadPoolExecutor() as executor:
        tasks = []
        for _ in range(len(car_msg)):
            selected_node = select_node(local_nodes)
            task = loop.run_in_executor(executor, selected_node.mine_block_dag)
            tasks.append(task)
        await asyncio.gather(*tasks)
    return local_blockchain

def perform_tampering_test(blockchain: Blockchain, num_tests: int):
    success_count = 0
    for _ in range(num_tests):
        if blockchain.tamper_chain():
            success_count += 1
    return success_count / num_tests

def generate_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    return private_key, public_key

async def main():
    # Generate keys for the nodes
    private_key1, public_key1 = generate_keys()
    private_key2, public_key2 = generate_keys()
    private_key3, public_key3 = generate_keys()
    private_key4, public_key4 = generate_keys()
    private_key5, public_key5 = generate_keys()
    private_key6, public_key6 = generate_keys()
    car_msg = [1 * 1000 * 1000, 6.4 * 1000 * 1000, 64 * 1000 * 1000, 130, 5, 5, 50]

    num_tests1 = 10000
    pow = []
    pos = []
    poh = []
    for num_tests in range(1, num_tests1 + 1):
        # Simulate PoW
        pow_blockchain = simulate_pow(Blockchain(), car_msg, private_key1)
        pow_tamper_success_rate = perform_tampering_test(pow_blockchain, num_tests)
        pow.append(pow_tamper_success_rate)
        #print(f"Tampering success rate for PoW: {pow_tamper_success_rate * 100}%")

        # Simulate PoS
        pos_blockchain = Blockchain()
        pos_nodes = [
            Node(1, pos_blockchain, 10, car_msg[0], private_key1, public_key1),
            Node(2, pos_blockchain, 30, car_msg[1], private_key2, public_key2),
            Node(3, pos_blockchain, 60, car_msg[2], private_key3, public_key3),
            Node(4, pos_blockchain, 10, car_msg[3], private_key4, public_key4),
            Node(5, pos_blockchain, 30, car_msg[4], private_key5, public_key5),
            Node(6, pos_blockchain, 60, car_msg[5], private_key6, public_key6)
        ]
        pos_blockchain = simulate_pos(pos_blockchain, pos_nodes, car_msg)
        pos_tamper_success_rate = perform_tampering_test(pos_blockchain, num_tests)
        pos.append(pos_tamper_success_rate)
        #print(f"Tampering success rate for PoS: {pos_tamper_success_rate * 100}%")

        # Simulate hybrid consensus with DAG structure
        hybrid_blockchain = Blockchain()
        hybrid_nodes = [
            Node(1, hybrid_blockchain, 10, car_msg[0], private_key1, public_key1),
            Node(2, hybrid_blockchain, 20, car_msg[1], private_key2, public_key2),
            Node(3, hybrid_blockchain, 30, car_msg[2], private_key3, public_key3),
            Node(4, hybrid_blockchain, 40, car_msg[3], private_key4, public_key4),
            Node(5, hybrid_blockchain, 50, car_msg[4], private_key5, public_key5),
            Node(6, hybrid_blockchain, 60, car_msg[5], private_key6, public_key6)
        ]
        hybrid_blockchain = await simulate_hybrid(hybrid_blockchain, hybrid_nodes, car_msg)
        hybrid_tamper_success_rate = perform_tampering_test(hybrid_blockchain, num_tests)
        poh.append(hybrid_tamper_success_rate)
        #print(f"Tampering success rate for Hybrid DAG: {hybrid_tamper_success_rate * 100}%")

    # Plotting the results
    consensus_mechanisms = ['PoW', 'PoS', 'Hybrid DAG']
    tamper_success_rates = [pow, pos, poh]
    print(sum(tamper_success_rates[0]) / 10000, sum(tamper_success_rates[1]) / 10000, sum(tamper_success_rates[2]) / 10000)
    plt.plot(tamper_success_rates[0], color='blue', linestyle='-', label='PoW Consensus')
    plt.plot(tamper_success_rates[1], color='green', linestyle='-', label='Pos Consensus')
    plt.plot(tamper_success_rates[2], color='red', linestyle='-', label='Poh Consensus')
    plt.xlabel('Consensus Mechanism')
    plt.ylabel('Tampering Success Rate (%)')
    plt.title('Comparison of Tampering Success Rates')
    plt.show()

    np.savetxt("avg_pow_Tampering Success Rate .txt", pow)
    np.savetxt("avg_pos_Tampering Success Rate .txt", pos)
    np.savetxt("avg_asyncio_hybrid_Tampering Success Rate .txt", poh)

if __name__ == "__main__":
    asyncio.run(main())

