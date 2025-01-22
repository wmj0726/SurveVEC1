import hashlib
import math
import numpy as np
import json
import time
import env
import rsa
from authentication import Authentication

# 定义区块链类
class Blockchain:
    def __init__(self):
        self.chain = []  # 区块链列表，存储所有的区块
        self.current_messages = []  # 当前未打包的消息列表

        self.add_genesis_block()  # 添加创世区块


    def add_genesis_block(self):
        # 创建创世区块
        block = {
            'index': 1,
            'timestamp': time.time(),
            'messages': [],
            'proof': 1,
            'previous_hash': 'null',
        }
        self.chain.append(block)

    def add_message(self, message, private_key, publick_key):
        msg = [message.id, message.state, message.message, message.reward]
        # 将一条消息添加到当前未打包的消息列表中
        # ,将私钥以及消息生成数字签名并通过公钥验证有效性
        signature = Authentication.sign_data(msg, private_key)
        try:
            Authentication.verify_signature(msg, signature, publick_key)
        except:
            print('Signature is not valid')
        self.current_messages.append(msg)  # 将消息加入到当前未打包的消息列表中

    def add_block(self, proof):
        # 创建一个新的区块，并将其加入到区块链中
        previous_hash = self.hash(self.chain[-1])  # 计算新区块的上一个哈希值
        block = {
            'index': len(self.chain) + 1,  # 区块索引
            'timestamp': time.time(),  # 区块时间戳
            'messages': self.current_messages,  # 当前区块包含的消息列表
            'proof': proof,  # 工作证明
            #'hash': self.hash(self.chain),
            'previous_hash': previous_hash,  # 上一个区块的哈希值
        }

        self.current_messages = []  # 清空当前未打包的消息列表
        self.chain.append(block)  # 将新的区块加入到区块链中

    def last_block(self):  # 获取当前链中最后一个区块
        return self.chain[-1]

    def vaild_chain(self) -> bool:
        """验证链是否合理：最长且有效
        Args:
            chain (List[Dict[str, Any]]): 传入链
        Returns:
            bool: 返回是否有效
        """
        chain = self.chain
        last_block = chain[0]  # 从第一个创世区块开始遍历验证
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            # 如果当前区块的前哈希和前一个计算出来的哈希值不同则是无效链
            if block['previous_hash'] != self.hash(last_block):
                print(block['previous_hash'], self.hash(last_block))
                print('区块hash不匹配，无效链')
                return False

            # 检验工作量证明是否符合要求
            if not self.valid_proof(last_block['proof'], block['proof']):
                print('工作量证明不匹配，无效链')
                return False

            last_block = block
            current_index += 1

        return True

    def proof_of_work(self, last_proof):
        self.vaild_chain()
        # 工作量证明算法
        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1
        return proof

    # valid_proof() 方法用于验证工作证明是否正确，其中参数 last_proof 是上一个区块的工作证明，参数 proof 是当前区块的工作证明。
    # 验证的方法是将上一个区块的工作证明和当前区块的工作证明拼接成一个字符串，并将其编码成字节串。然后对该字节串计算 SHA256 哈希值，取其前四个字符，判断是否为 "0000"。如果是，则说明工作证明正确；否则不正确。

    @staticmethod
    def valid_proof(last_proof, proof):
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == '0000'


    # hash() 方法用于计算区块的 SHA256 哈希值，其中参数 block 是待计算哈希值的区块。
    # 首先将该区块转换为 JSON 格式的字符串，并将其编码成字节串。然后对该字节串计算 SHA256 哈希值，返回哈希值的十六进制字符串表示。

    @staticmethod
    def hash(block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def create_block(self, transaction, proof):
        # 创建区块并将交易写入区块链
        previous_hash = self.hash(self.chain[-1])  # 计算新区块的上一个哈希值
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),  # 区块时间戳
            'Transaction': transaction,
            'proof': proof,  # 工作证明
            'previous_hash': previous_hash,
            # 其他区块信息
        }
        self.chain.append(block)  # 将新的区块加入到区块链中

    def update_reputation(self, car_id, new_reputation):
        # 更新智能汽车的信誉值
        blockchain = env.blockchain
        if blockchain.chain[-1]['Transaction']['Car_ID'] == car_id:
            blockchain.chain[-1]['Transaction'][f'{car_id}_Reputation'] = new_reputation

    def consensus(self):
        # 共识机制，验证区块链的完整性和一致性
        pass

# 哈希函数
def calculate_hash(data):
    data_str = str(data)
    return hashlib.sha256(data_str.encode()).hexdigest()

