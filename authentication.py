import rsa

class Authentication:
    @staticmethod
    def generate_key_pair():
        publicKey, privateKey = rsa.newkeys(512)
        return publicKey, privateKey

    @staticmethod
    def sign_data(data, privateKey):
        data = str(data).encode()
        signature = rsa.sign(data, privateKey, 'SHA-256')
        return signature

    @staticmethod
    def verify_signature(data, signature, publicKey):
        data = str(data).encode()
        try:
            rsa.verify(data, signature, publicKey)
            return True
        except:
            return False